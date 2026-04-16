#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ascend_prof_tool.py — AscendProfTool 单卡/多卡 trace 生成验证

参考 tests/python/deepep/test_intranode_direct.py 的测试框架，使用:
  - VecAddProf 算子 (单卡场景)
  - NotifyDispatchProf 算子 (多卡场景)

验证 AscendProfTool 的完整流程:
  1. 时钟校准 (单卡 / 多卡)
  2. 算子执行 + 打点 buffer 解析
  3. Trace Event Format JSON 生成

Usage:
    # 单卡测试 (无需多进程):
    python test_ascend_prof_tool.py --mode single

    # 多卡测试 (使用 torch.multiprocessing.spawn):
    python test_ascend_prof_tool.py --mode multi --num-processes 2

    # 全部测试:
    python test_ascend_prof_tool.py --mode all --num-processes 2

    # 仅模拟 (无 NPU, 验证解析 + JSON 生成逻辑):
    python test_ascend_prof_tool.py --mode mock
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory so we can import ascend_prof_tool
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ascend_prof_tool import AscendProfTool, CYCLES_PER_US, PROF_ITER_END_TAG


# =========================================================================== #
#                           常量
# =========================================================================== #

BLOCK_DIM = 8
TILE_LENGTH = 256
PROF_ITER_CORE_STRIDE = 136  # from kernel_tool.h: CORE_META_SIZE(8) + CORE_DATA_SIZE(128)
PROF_CORE_HEADER_SIZE = 8    # from kernel_tool.h: CORE_HEADER_SIZE
PROF_GLOBAL_HEADER_SIZE = 8  # from kernel_tool.h: GLOBAL_HEADER_SIZE
PROF_MAX_SLOTS = 64
PROF_MAX_ITERS = 10
PROF_BLOCK_DIM_FALLBACK = 64
SEND_PER_GROUP = 3  # notify_dispatch sendData 组: (token_count, prefix_sum, num_tokens)

NOTIFY_DISPATCH_TAG_NAMES = {
    0: "Init",
    100: "AssembleSendDataDone",
    200: "InputToShareSliceDone",
    300: "ShareToShareSliceDone",
    400: "ReorderOutputDone",
    500: "BuildTotalRecvTokensDone",
    600: "BuildRecvCountDone",
    700: "BuildRecvOffsetDone",
    800: "BuildMaxBsDone",
    900: "BuildRecvTokenPerExpDone",
    999: "AllDone",
    PROF_ITER_END_TAG: "IterEnd",
}


# =========================================================================== #
#                    辅助函数: 分布式初始化
# =========================================================================== #

def init_dist(local_rank: int, num_local_ranks: int):
    """初始化分布式环境 (参考 deepep test_intranode_direct.py)."""
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "29500"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    global_rank = node_rank * num_local_ranks + local_rank
    world_size = num_nodes * num_local_ranks

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=global_rank,
    )

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    group = dist.new_group(list(range(world_size)))

    return dist.get_rank(), dist.get_world_size(), group


# =========================================================================== #
#                    辅助函数: profiling buffer 大小
# =========================================================================== #

def prof_buf_int64_size(block_dim: int, max_iters: int = 1) -> int:
    """kernel_tool.h profiling buffer 所需 int64 元素数 (多轮迭代)."""
    global_header = PROF_GLOBAL_HEADER_SIZE
    core_headers = block_dim * PROF_CORE_HEADER_SIZE
    data_region = max_iters * block_dim * PROF_ITER_CORE_STRIDE
    return global_header + core_headers + data_region


def infer_block_dim_from_offsets(rank_offsets, rank, fallback=PROF_BLOCK_DIM_FALLBACK):
    """从 calibrate_rank_clocks 的结果推断当前 rank 实际核数."""
    core_ids = [core for (r, core) in rank_offsets.keys() if r == rank]
    if not core_ids:
        return fallback
    return max(core_ids) + 1


def compute_notify_output_shapes(send_count: int, rank_size: int) -> dict:
    """根据 sendCount/rankSize 推导 NotifyDispatchProf 输出 shape."""
    num_experts = send_count // SEND_PER_GROUP
    return {
        "sendDataOffset": (num_experts,),
        "recvData": (send_count,),
        "totalRecvTokens": (1,),
        "recvCount": (num_experts,),
        "recvOffset": (num_experts,),
        "maxBs": (1,),
        "recvTokensPerExpert": (num_experts // rank_size,),
    }


def validate_notify_output_shapes(outputs: dict, expected_shapes: dict) -> bool:
    """校验 NotifyDispatchProf 输出 shape."""
    all_pass = True
    for name, expected in expected_shapes.items():
        actual = outputs.get(name)
        if actual is None:
            print(f"  [FAIL] {name}: missing", flush=True)
            all_pass = False
            continue
        if actual != expected:
            print(f"  [FAIL] {name}: expected {expected}, got {actual}", flush=True)
            all_pass = False
        else:
            print(f"  [PASS] {name}: shape = {actual}", flush=True)
    return all_pass


# =========================================================================== #
#                    辅助函数: trace JSON 校验
# =========================================================================== #

def validate_trace_json(path: str, expected_kernels: int, expected_cores: int,
                        min_events: int = 1) -> bool:
    """读取并校验生成的 trace JSON 文件."""
    if not os.path.exists(path):
        print(f"  [FAIL] Trace file not found: {path}")
        return False

    with open(path, "r") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    if len(events) < min_events:
        print(f"  [FAIL] Too few events: {len(events)} (expected >= {min_events})")
        return False

    # 检查 metadata 事件
    process_names = [e for e in events if e.get("name") == "process_name"]
    thread_names = [e for e in events if e.get("name") == "thread_name"]
    duration_events = [e for e in events if e.get("ph") == "X"]

    # pid = rank_id, tid = core_id
    core_ids = set(e["tid"] for e in thread_names)
    rank_ids = set(e["pid"] for e in process_names)

    ok = True
    # 检查 kernel/iteration 数量: 每个 iteration 的每个 core 贡献 duration events,
    # expected_kernels 实际代表预期的 iteration 总数 (跨所有 prof_data)
    kernel_cats = set(e.get("cat", "") for e in duration_events)
    iter_ids = set((e.get("cat", ""), e.get("args", {}).get("iter", -1))
                   for e in duration_events)
    actual_iters = len(iter_ids)
    if actual_iters != expected_kernels:
        print(f"  [FAIL] Expected {expected_kernels} iterations, "
              f"got {actual_iters} (kernels: {kernel_cats})")
        ok = False
    if len(core_ids) < expected_cores:
        print(f"  [WARN] Expected {expected_cores} cores, got {len(core_ids)}")
        ok = False
    if len(duration_events) == 0:
        print(f"  [FAIL] No duration events found")
        ok = False

    # 检查时间戳合理性 (不应为负数或 NaN)
    for e in duration_events:
        ts = e.get("ts", 0)
        dur = e.get("dur", 0)
        if dur < 0:
            print(f"  [FAIL] Negative duration: {dur} in event {e.get('name')}")
            ok = False
            break

    if ok:
        print(f"  [PASS] {path}: {len(events)} events, "
              f"{len(core_ids)} cores, "
              f"{len(duration_events)} duration events")
    return ok


# =========================================================================== #
#                    Mock 测试 (无需 NPU)
# =========================================================================== #

def build_mock_prof_data(block_dim: int, tile_num: int, num_iters: int = 1) -> dict:
    """构造模拟的 kernel_tool.h profiling buffer 并解析 (支持多轮迭代)."""
    global_header_size = PROF_GLOBAL_HEADER_SIZE
    core_header_size = PROF_CORE_HEADER_SIZE
    core_meta_size = 8
    iter_core_stride = PROF_ITER_CORE_STRIDE

    core_header_region_start = global_header_size
    data_region_start = global_header_size + block_dim * core_header_size

    buf_size = data_region_start + num_iters * block_dim * iter_core_stride
    buf = np.zeros(buf_size, dtype=np.int64)

    # Global Header
    buf[0] = block_dim
    buf[1] = PROF_MAX_SLOTS
    buf[2] = num_iters
    buf[3] = core_header_size
    buf[4] = core_meta_size
    buf[5] = iter_core_stride
    buf[6] = core_header_region_start
    buf[7] = data_region_start

    # Per-core headers: iteration count
    for core in range(block_dim):
        hdr_base = core_header_region_start + core * core_header_size
        buf[hdr_base] = num_iters

    base_cycle = 1_000_000
    for it in range(num_iters):
        for core in range(block_dim):
            base = (data_region_start
                    + it * block_dim * iter_core_stride
                    + core * iter_core_stride)
            cycle = base_cycle + it * 10000 + core * 50  # 模拟核间微小偏差

            events = [(0, cycle)]
            for t in range(tile_num):
                cycle += 100
                events.append((100 + t, cycle))
                cycle += 200
                events.append((200 + t, cycle))
                cycle += 50
                events.append((300 + t, cycle))
            cycle += 10
            events.append((PROF_ITER_END_TAG, cycle))

            buf[base] = len(events)
            for j, (tag, ts) in enumerate(events):
                buf[base + core_meta_size + j * 2] = tag
                buf[base + core_meta_size + j * 2 + 1] = ts

    return AscendProfTool.parse_prof_buf(buf)


def test_mock(trace_dir: str):
    """
    Mock 测试: 验证 AscendProfTool 的解析和 JSON 生成逻辑, 无需 NPU.

    场景 1: 单卡 — VecAddProf mock data
    场景 2: 单卡多 kernel (VecAddProf × 2)
    场景 3: 单卡多轮迭代 (VecAddProf × 3 iters)
    """
    os.makedirs(trace_dir, exist_ok=True)
    tool = AscendProfTool(sync_rounds=16)
    all_pass = True

    # ---- 场景 1: 单卡 VecAddProf ----
    print("\n[Mock] 场景 1: 单卡 VecAddProf trace 生成")
    prof_data = build_mock_prof_data(block_dim=BLOCK_DIM, tile_num=4)
    assert prof_data["block_num"] == BLOCK_DIM
    assert len(prof_data["iterations"]) == 1
    assert len(prof_data["iterations"][0]["cores"]) == BLOCK_DIM

    # 模拟核偏移 (core 0 为参考)
    offsets = {i: float(i * 50) for i in range(BLOCK_DIM)}
    offsets[0] = 0.0

    trace_path = os.path.join(trace_dir, "mock_single_card.json")
    tool.generate_trace_json(
        prof_data_list=[prof_data],
        offsets=offsets,
        output_path=trace_path,
        kernel_names={0: "VecAddProf"},
    )
    if not validate_trace_json(trace_path, expected_kernels=1, expected_cores=BLOCK_DIM):
        all_pass = False

    # ---- 场景 2: 单卡多 kernel (VecAddProf × 2) ----
    print("\n[Mock] 场景 2: 单卡多 kernel trace 生成")
    prof_data_1 = build_mock_prof_data(block_dim=BLOCK_DIM, tile_num=4)
    prof_data_2 = build_mock_prof_data(block_dim=BLOCK_DIM, tile_num=2)

    trace_path = os.path.join(trace_dir, "mock_multi_kernel.json")
    tool.generate_trace_json(
        prof_data_list=[prof_data_1, prof_data_2],
        offsets=offsets,
        output_path=trace_path,
        kernel_names={0: "VecAddProf_Run1", 1: "VecAddProf_Run2"},
    )
    if not validate_trace_json(trace_path, expected_kernels=2, expected_cores=BLOCK_DIM):
        all_pass = False

    # ---- 场景 3: 单卡多轮迭代 (VecAddProf × 3 iters) ----
    print("\n[Mock] 场景 3: 单卡多轮迭代 trace 生成 (3 iterations)")
    prof_data_multi_iter = build_mock_prof_data(block_dim=BLOCK_DIM, tile_num=4, num_iters=3)
    assert prof_data_multi_iter["block_num"] == BLOCK_DIM
    assert len(prof_data_multi_iter["iterations"]) == 3

    trace_path = os.path.join(trace_dir, "mock_multi_iter.json")
    tool.generate_trace_json(
        prof_data_list=[prof_data_multi_iter],
        offsets=offsets,
        output_path=trace_path,
        kernel_names={0: "VecAddProf"},
    )
    if not validate_trace_json(trace_path, expected_kernels=3, expected_cores=BLOCK_DIM):
        all_pass = False

    print(f"\n{'=' * 50}")
    print(f"[Mock] {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 50}")
    return all_pass


# =========================================================================== #
#                    单卡测试 (VecAddProf on NPU)
# =========================================================================== #

def test_single_card(trace_dir: str):
    """
    单卡测试: VecAddProf 算子 + AscendProfTool trace 生成.

    用一个 prof_buf (max_iters=NUM_RUNS) 承载多次 VecAddProf 算子调用的
    打点数据, 验证跨 launch 追加写入能力 (PROF_INIT 从 GM 读 iterCount).

    流程:
      1. calibrate_core_clocks() — 单卡多核时钟校准
      2. 运行 VecAddProf × NUM_RUNS (共享同一 prof_buf)
      3. parse_prof_buf() — 一次解析, 得到 NUM_RUNS 轮迭代
      4. generate_trace_json() — 生成包含所有迭代的 trace JSON
      5. 校验 JSON 结构
    """
    import torch
    import torch_npu  # noqa: F401
    import ascend_tool

    os.makedirs(trace_dir, exist_ok=True)
    tool = AscendProfTool(sync_rounds=16)
    all_pass = True
    NUM_RUNS = 10
    assert NUM_RUNS <= PROF_MAX_ITERS, (
        f"NUM_RUNS={NUM_RUNS} exceeds kernel PROF_MAX_ITERS={PROF_MAX_ITERS}; "
        "increase PROF_MAX_ITERS in kernel_tool.h or reduce NUM_RUNS"
    )

    print("\n[Single] Step 1: 单卡多核时钟校准")
    offsets = tool.calibrate_core_clocks()
    print(f"  Core offsets (cycles): { {k: f'{v:.1f}' for k, v in offsets.items()} }")
    assert 0 in offsets and offsets[0] == 0.0, "Core 0 offset should be 0"

    print(f"\n[Single] Step 2: 运行 VecAddProf × {NUM_RUNS} "
          f"(共享 prof_buf, max_iters={NUM_RUNS})")
    N = BLOCK_DIM * TILE_LENGTH * 4
    prof_size = prof_buf_int64_size(BLOCK_DIM, max_iters=NUM_RUNS)
    prof_buf = torch.zeros(prof_size, dtype=torch.int64, device="npu")
    x = torch.randn(N, dtype=torch.float16, device="npu")
    y = torch.randn(N, dtype=torch.float16, device="npu")

    for run_idx in range(NUM_RUNS):
        z = ascend_tool.vec_add_prof(x, y, prof_buf)

    torch.npu.synchronize()

    # 正确性验证
    z_cpu = z.cpu().float()
    ref = x.cpu().float() + y.cpu().float()
    max_err = (z_cpu - ref).abs().max().item()
    status = 'PASS' if max_err < 0.01 else 'FAIL'
    print(f"  VecAdd correctness: max_error = {max_err:.6f} ({status})")
    if max_err >= 0.01:
        all_pass = False

    print(f"\n[Single] Step 3: 解析打点 buffer ({NUM_RUNS} iterations)")
    prof_data = AscendProfTool.parse_prof_buf(prof_buf)
    print(f"  block_num={prof_data['block_num']}, "
          f"iterations={len(prof_data['iterations'])}")
    assert len(prof_data["iterations"]) == NUM_RUNS, \
        f"Expected {NUM_RUNS} iterations, got {len(prof_data['iterations'])}"

    for it_idx, it_data in enumerate(prof_data["iterations"]):
        cores = it_data["cores"]
        active = sum(1 for c in cores if c['count'] > 0)
        print(f"  Iter {it_idx}: cores with data = {active}")

    print("\n[Single] Step 4: 生成 trace JSON")
    trace_path = os.path.join(trace_dir, "single_card_vec_add.json")
    tool.generate_trace_json(
        prof_data_list=[prof_data],
        offsets=offsets,
        output_path=trace_path,
        kernel_names={0: "VecAddProf"},
    )
    if not validate_trace_json(trace_path, expected_kernels=NUM_RUNS,
                               expected_cores=BLOCK_DIM):
        all_pass = False

    print(f"\n{'=' * 50}")
    print(f"[Single] {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 50}")
    return all_pass


# =========================================================================== #
#                    多卡测试 (NotifyDispatchProf)
# =========================================================================== #

def test_multi_card_worker(
    local_rank: int,
    num_local_ranks: int,
    trace_dir: str,
):
    """多卡测试的 worker 进程入口."""
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    import ascend_tool

    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    print(f"[Rank {rank}] Initialized (local_rank={local_rank}, "
          f"world_size={world_size})", flush=True)

    os.makedirs(trace_dir, exist_ok=True)
    tool = AscendProfTool(sync_rounds=16)
    all_pass = True
    NUM_RUNS = 10

    # ================================================================
    # 场景 1: 多卡时钟校准
    # ================================================================
    print(f"\n[Rank {rank}] 场景 1: 多卡时钟校准", flush=True)

    # 预热: 在自定义算子调用前，先在同一 group 上执行一次标准 HCCL 操作，
    # 确保 HCCL 窗口内存已被框架分配和注册.
    dist.barrier(group=group)
    warmup_tensor = torch.ones(1, dtype=torch.int64, device="npu")
    dist.all_reduce(warmup_tensor, op=dist.ReduceOp.SUM, group=group)
    torch.npu.synchronize()

    rank_offsets = tool.calibrate_rank_clocks(world_size, rank, group=group)
    if rank == 0:
        print(f"  Global offsets sample: "
              f"(0,0)={rank_offsets.get((0,0), 'N/A'):.1f}, "
              f"(0,1)={rank_offsets.get((0,1), 'N/A'):.1f}, "
              f"(1,0)={rank_offsets.get((1,0), 'N/A'):.1f}",
              flush=True)

    dist.barrier()

    # ================================================================
    # 场景 2: 多卡 NotifyDispatchProf (共享 prof_buf, 多轮 launch)
    # ================================================================
    print(f"\n[Rank {rank}] 场景 2: 多卡 NotifyDispatchProf trace "
          f"(共享 prof_buf, NUM_RUNS={NUM_RUNS})", flush=True)

    backend = group._get_backend(torch.device("npu"))
    comm_group = backend.get_hccl_comm_name(rank)
    if isinstance(comm_group, bytes):
        comm_group = comm_group.decode("utf-8")

    num_experts = world_size * 4  # 每 rank 4 个 expert
    num_tokens = 1024
    send_count = num_experts * SEND_PER_GROUP

    rng = np.random.default_rng(42 + rank)
    token_per_expert_np = rng.integers(
        1, max(2, num_tokens // num_experts * 3),
        size=num_experts, dtype=np.int32
    )
    token_per_expert_np = (
        token_per_expert_np / token_per_expert_np.sum() * num_tokens
    ).astype(np.int32)
    diff = num_tokens - token_per_expert_np.sum()
    token_per_expert_np[0] += diff
    assert token_per_expert_np.sum() == num_tokens, \
        f"Token sum mismatch: {token_per_expert_np.sum()} != {num_tokens}"

    token_per_expert_data = torch.from_numpy(token_per_expert_np).to(
        dtype=torch.int32, device="npu"
    )
    send_data = torch.zeros(send_count, dtype=torch.int32, device="npu")

    prof_block_dim = infer_block_dim_from_offsets(rank_offsets, rank)
    prof_size = prof_buf_int64_size(prof_block_dim, max_iters=NUM_RUNS)
    prof_buf = torch.zeros(prof_size, dtype=torch.int64, device="npu")
    print(f"  [Rank {rank}] prof_buf elems={prof_size} "
          f"(block_dim={prof_block_dim}, max_iters={NUM_RUNS}, "
          f"max_slots={PROF_MAX_SLOTS})", flush=True)

    # 通过 HCCL allreduce 建立 device 端依赖，缩小跨 rank launch 抖动
    sync_tensor = torch.ones(1, dtype=torch.int64, device="npu")
    dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM, group=group)
    send_data[:1].copy_(sync_tensor.to(torch.int32))

    output_names = [
        "sendDataOffset", "recvData", "totalRecvTokens",
        "recvCount", "recvOffset", "maxBs", "recvTokensPerExpert"
    ]
    outputs = None
    for run_idx in range(NUM_RUNS):
        outputs = ascend_tool.notify_dispatch_prof(
            send_data=send_data,
            token_per_expert_data=token_per_expert_data,
            prof_buf=prof_buf,
            send_count=send_count,
            num_tokens=num_tokens,
            comm_group=comm_group,
            rank_size=world_size,
            rank_id=rank,
            local_rank_size=num_local_ranks,
            local_rank_id=local_rank,
        )

    torch.npu.synchronize()
    dist.barrier(group=group)

    output_dict = {name: tuple(t.shape) for name, t in zip(output_names, outputs)}
    expected_shapes = compute_notify_output_shapes(send_count, world_size)
    shape_pass = validate_notify_output_shapes(output_dict, expected_shapes)
    all_pass = all_pass and shape_pass

    if rank == 0:
        total_recv = int(outputs[2].cpu().item())
        max_bs = int(outputs[5].cpu().item())
        consistency_pass = (total_recv > 0) and (max_bs > 0)
        print(f"  [Rank 0] totalRecvTokens={total_recv}, maxBs={max_bs}, "
              f"consistency={'PASS' if consistency_pass else 'FAIL'}", flush=True)
        all_pass = all_pass and consistency_pass

    notify_data = AscendProfTool.parse_prof_buf(prof_buf)
    print(f"  [Rank {rank}] block_num={notify_data['block_num']}, "
          f"iterations={len(notify_data['iterations'])}", flush=True)
    assert len(notify_data["iterations"]) == NUM_RUNS, \
        f"Expected {NUM_RUNS} iterations, got {len(notify_data['iterations'])}"

    trace_path = os.path.join(trace_dir, f"multi_card_notify_dispatch_rank{rank}.json")
    tool.generate_trace_json(
        prof_data_list=[notify_data],
        offsets=rank_offsets,
        output_path=trace_path,
        kernel_names={0: "NotifyDispatchProf"},
        rank_id=rank,
        tag_names=NOTIFY_DISPATCH_TAG_NAMES,
    )
    if not validate_trace_json(trace_path, expected_kernels=NUM_RUNS,
                               expected_cores=prof_block_dim):
        all_pass = False

    # ================================================================
    # 场景 3: 合并多 rank trace (rank 0 收集所有 rank 数据)
    # ================================================================
    dist.barrier()
    print(f"\n[Rank {rank}] 场景 3: 合并多 rank trace", flush=True)

    merged_path = os.path.join(trace_dir, "multi_card_merged.json")
    tool.generate_merged_trace_json(
        prof_data_list=[notify_data],
        offsets=rank_offsets,
        output_path=merged_path,
        world_size=world_size,
        rank=rank,
        kernel_names={0: "NotifyDispatchProf"},
        tag_names=NOTIFY_DISPATCH_TAG_NAMES,
    )

    if rank == 0:
        if not validate_trace_json(merged_path, expected_kernels=NUM_RUNS,
                                   expected_cores=prof_block_dim):
            all_pass = False

    # ================================================================
    # 汇总
    # ================================================================
    dist.barrier()

    # 通过 allreduce 传播失败状态
    pass_tensor = torch.tensor([1 if all_pass else 0], dtype=torch.int64,
                               device="npu")
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    all_pass = pass_tensor.item() == 1

    if rank == 0:
        print(f"\n{'=' * 50}")
        print(f"[Multi] {'ALL PASSED' if all_pass else 'SOME FAILED'}")
        print(f"{'=' * 50}", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if not all_pass:
        raise RuntimeError(f"[Rank {rank}] Multi-card test FAILED")


# =========================================================================== #
#                           入口
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="AscendProfTool 单卡/多卡 trace 生成验证"
    )
    parser.add_argument(
        "--mode", type=str, default="mock",
        choices=["mock", "single", "multi", "all"],
        help="测试模式: mock (无NPU), single (单卡), multi (多卡), all (全部)",
    )
    parser.add_argument(
        "--num-processes", type=int, default=2,
        help="多卡测试进程数 (default: 2)",
    )
    parser.add_argument(
        "--trace-dir", type=str, default="./traces",
        help="Trace 文件输出目录 (default: ./traces)",
    )
    args = parser.parse_args()

    if args.mode in ("mock", "all"):
        print("\n" + "=" * 60)
        print("  Mock 测试 (无需 NPU)")
        print("=" * 60)
        test_mock(args.trace_dir)

    if args.mode in ("single", "all"):
        print("\n" + "=" * 60)
        print("  单卡测试 (VecAddProf)")
        print("=" * 60)
        test_single_card(args.trace_dir)

    if args.mode in ("multi", "all"):
        print("\n" + "=" * 60)
        print(f"  多卡测试 ({args.num_processes} ranks)")
        print("=" * 60)
        import torch
        torch.multiprocessing.spawn(
            fn=test_multi_card_worker,
            args=(args.num_processes, args.trace_dir),
            nprocs=args.num_processes,
        )


if __name__ == "__main__":
    main()
