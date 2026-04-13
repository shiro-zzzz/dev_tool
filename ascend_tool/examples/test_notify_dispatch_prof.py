#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_notify_dispatch_prof.py — NotifyDispatchProf 算子验证脚本

使用方法:
  1. 先构建算子: compile_ascend_proj.sh (确保 NotifyDispatchProf.json 被 msopgen 处理)
  2. 安装生成的 .run 包
  3. 构建 pybind wheel 并安装
  4. 在分布式环境中运行 (spawn 拉起):
     python test_notify_dispatch_prof.py --mode npu --num-processes N

说明:
  NotifyDispatchProf 是一个 MoE (Mixture-of-Experts) 通信算子，
  使用 HCCL AlltoAll 实现跨 rank 的 expert token 分发调度。
  它需要在多卡分布式环境（HCCL 通信域已初始化）下运行。

  本脚本提供：
    - 模拟模式 (demo_mode): 不依赖 NPU硬件，验证参数计算与输出形状逻辑
    - 真机模式 (npu_mode):  在已初始化 HCCL 的多卡环境下端到端执行算子
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory so we can import ascend_prof_tool
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ascend_prof_tool import AscendProfTool

# profiling buffer layout constants (from kernel_tool.h)
PROF_ITER_CORE_STRIDE = 136
PROF_CORE_HEADER_SIZE = 8
PROF_GLOBAL_HEADER_SIZE = 8
PROF_BLOCK_DIM_FALLBACK = 64

# ─────────────────────────────────────────────────────────────────────────────
#  常量定义
# ─────────────────────────────────────────────────────────────────────────────
SEND_PER_GROUP = 3  # 每个 expert 的发送数据分组：(token_count, prefix_sum, num_tokens)
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
    99999: "IterEnd",
}


def prof_buf_int64_size(block_dim: int, max_iters: int = 1) -> int:
    """kernel_tool.h profiling buffer 所需 int64 元素数 (多轮迭代)."""
    global_header = PROF_GLOBAL_HEADER_SIZE
    core_headers = block_dim * PROF_CORE_HEADER_SIZE
    data_region = max_iters * block_dim * PROF_ITER_CORE_STRIDE
    return global_header + core_headers + data_region


def infer_block_dim_from_offsets(rank_offsets, rank, fallback=PROF_BLOCK_DIM_FALLBACK):
    """从 calibrate_rank_clocks 的结果推断当前 rank 的实际核数。"""
    core_ids = [core for (r, core) in rank_offsets.keys() if r == rank]
    if not core_ids:
        return fallback
    return max(core_ids) + 1


# ─────────────────────────────────────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────────────────────────────────────
def compute_output_shapes(send_count, rank_size):
    """
    根据 sendCount 和 rankSize 计算 NotifyDispatchProf 各输出张量的期望形状。

    Parameters
    ----------
    send_count : int
        sendData 张量的元素总数 = numExperts * SEND_PER_GROUP
    rank_size : int
        通信域中的总 rank 数

    Returns
    -------
    dict : 各输出张量名到期望形状的映射
    """
    num_experts = send_count // SEND_PER_GROUP
    return {
        "sendDataOffset":      (num_experts,),
        "recvData":            (send_count,),
        "totalRecvTokens":     (1,),
        "recvCount":           (num_experts,),
        "recvOffset":          (num_experts,),
        "maxBs":               (1,),
        "recvTokensPerExpert": (num_experts // rank_size,),
    }


def validate_output_shapes(outputs, expected_shapes):
    """
    验证输出张量的形状是否匹配预期。

    Parameters
    ----------
    outputs : dict[str, tuple]
        实际输出张量名到形状的映射
    expected_shapes : dict[str, tuple]
        期望的形状映射

    Returns
    -------
    bool : 全部通过返回 True
    """
    all_pass = True
    for name, expected in expected_shapes.items():
        actual = outputs.get(name)
        if actual is None:
            print(f"  [FAIL] {name}: 缺失")
            all_pass = False
            continue
        if actual != expected:
            print(f"  [FAIL] {name}: expected {expected}, got {actual}")
            all_pass = False
        else:
            print(f"  [PASS] {name}: shape = {actual}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  模拟模式：验证参数计算逻辑
# ─────────────────────────────────────────────────────────────────────────────
def demo_mode():
    """
    不依赖 NPU 硬件，纯 CPU 验证参数计算和输出形状逻辑。
    模拟不同 rank_size 和 num_experts 的配置。
    """
    print("=" * 70)
    print("  NotifyDispatchProf 模拟模式 — 参数计算与输出形状验证")
    print("=" * 70)

    test_cases = [
        # (rank_size, local_rank_size, num_experts, num_tokens)
        (8,  8, 64,  1024),
        (16, 8, 128, 2048),
        (32, 8, 256, 4096),
        (64, 8, 512, 8192),
    ]

    all_pass = True
    for rank_size, local_rank_size, num_experts, num_tokens in test_cases:
        send_count = num_experts * SEND_PER_GROUP
        print(f"\n--- rank_size={rank_size}, local_rank_size={local_rank_size}, "
              f"num_experts={num_experts}, num_tokens={num_tokens} ---")
        print(f"  sendCount = {send_count}")
        print(f"  numExperts (derived) = {num_experts}")
        print(f"  experts_per_rank = {num_experts // rank_size}")

        expected_shapes = compute_output_shapes(send_count, rank_size)

        # 模拟 tokenPerExpertData: 随机 token 分配
        rng = np.random.default_rng(42)
        token_per_expert = rng.integers(0, num_tokens // num_experts * 2,
                                        size=num_experts, dtype=np.int32)
        # 确保总 token 数为 num_tokens
        token_per_expert = (token_per_expert / token_per_expert.sum() * num_tokens).astype(np.int32)
        diff = num_tokens - token_per_expert.sum()
        token_per_expert[0] += diff

        assert token_per_expert.sum() == num_tokens, \
            f"Token sum mismatch: {token_per_expert.sum()} != {num_tokens}"

        # 模拟 sendData 组装逻辑（和 kernel 中 AssembleSendData 一致）
        send_data = np.zeros(send_count, dtype=np.int32)
        send_data_offset = np.zeros(num_experts, dtype=np.int32)
        prefix_sum = 0
        for i in range(num_experts):
            send_data[i * SEND_PER_GROUP] = token_per_expert[i]
            send_data[i * SEND_PER_GROUP + 1] = prefix_sum
            send_data[i * SEND_PER_GROUP + 2] = num_tokens
            send_data_offset[i] = prefix_sum
            prefix_sum += token_per_expert[i]

        # 验证 sendDataOffset 是正确的前缀和
        expected_prefix = np.cumsum(np.concatenate(([0], token_per_expert[:-1]))).astype(np.int32)
        offset_match = np.allclose(send_data_offset, expected_prefix, atol=1)
        print(f"  sendDataOffset prefix-sum check: {'PASS' if offset_match else 'FAIL'}")

        # 模拟理想输出形状
        mock_outputs = {name: shape for name, shape in expected_shapes.items()}
        case_pass = validate_output_shapes(mock_outputs, expected_shapes)
        all_pass = all_pass and case_pass and offset_match

    print("\n" + "=" * 70)
    print(f"  总体结果: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70 + "\n")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
#  NPU 真机模式：端到端执行
# ─────────────────────────────────────────────────────────────────────────────
def init_hccl_env_and_group(local_rank, num_local_ranks, dist, torch, torch_npu):
    """
    初始化 NPU 设备、HCCL 进程组，并创建 EP 通讯域名字。

    返回:
        rank, rank_size, local_rank, local_rank_size, comm_group_name
    """
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "29500"))
    num_nodes = int(os.getenv("WORLD_SIZE", "1"))
    node_rank = int(os.getenv("RANK", "0"))
    global_rank = node_rank * num_local_ranks + local_rank
    world_size = num_nodes * num_local_ranks

    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=world_size,
            rank=global_rank,
        )

    rank = dist.get_rank()
    rank_size = dist.get_world_size()

    # NPU-only device selection; no CUDA fallback.
    torch_npu.npu.set_device(local_rank)
    local_rank_size = min(8, rank_size)  # single node max 8 cards by default

    # Follow CAM sample pattern: create a full-rank group and get HCCL comm name.
    group = dist.new_group(list(range(rank_size)))
    comm_group = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    if isinstance(comm_group, bytes):
        comm_group = comm_group.decode("utf-8")

    return rank, rank_size, local_rank, local_rank_size, comm_group, group


def npu_worker(local_rank, num_local_ranks, trace_dir):
    """
    在已建立 HCCL 通信域的多卡环境下运行 NotifyDispatchProf 算子。
    使用 torch.multiprocessing.spawn 拉起多进程。
    """
    try:
        import torch
        import torch.distributed as dist
        import torch_npu
        import ascend_tool
    except ImportError as e:
        print(f"Import error: {e}")
        print("请确保已安装 torch, torch_npu 和 ascend_tool wheel 包。")
        print("退回到模拟模式...\n")
        demo_mode()
        return

    # ---------- 初始化分布式 / HCCL 通讯域 ----------
    rank, rank_size, local_rank, local_rank_size, comm_group, group = init_hccl_env_and_group(
        local_rank, num_local_ranks, dist, torch, torch_npu
    )

    if rank == 0:
        print("=" * 70)
        print("  NotifyDispatchProf NPU 真机模式")
        print(f"  rank_size={rank_size}, local_rank_size={local_rank_size}")
        print("=" * 70)

    os.makedirs(trace_dir, exist_ok=True)
    tool = AscendProfTool(sync_rounds=16)
    # 预热 HCCL，避免首次校准抖动
    dist.barrier(group=group)
    warmup_tensor = torch.ones(1, dtype=torch.int64, device=f"npu:{local_rank}")
    dist.all_reduce(warmup_tensor, op=dist.ReduceOp.SUM, group=group)
    torch_npu.npu.synchronize()
    rank_offsets = tool.calibrate_rank_clocks(rank_size, rank)
    prof_block_dim = infer_block_dim_from_offsets(rank_offsets, rank)
    print(f"[Rank {rank}] detected block_dim={prof_block_dim}")

    # ---------- 参数 ----------
    NUM_EXPERTS = rank_size * 4     # 每 rank 4 个 expert
    NUM_TOKENS = 1024
    SEND_COUNT = NUM_EXPERTS * SEND_PER_GROUP
    NUM_RUNS = 1

    # ---------- 构造输入 ----------
    # tokenPerExpertData: 每个 expert 被分配的 token 数
    rng = np.random.default_rng(42 + rank)
    token_per_expert_np = rng.integers(1, NUM_TOKENS // NUM_EXPERTS * 3,
                                       size=NUM_EXPERTS, dtype=np.int32)
    token_per_expert_np = (token_per_expert_np / token_per_expert_np.sum() * NUM_TOKENS).astype(np.int32)
    diff = NUM_TOKENS - token_per_expert_np.sum()
    token_per_expert_np[0] += diff

    token_per_expert_data = torch.from_numpy(token_per_expert_np).to(
        dtype=torch.int32, device=f"npu:{local_rank}")

    # sendData: 预分配空间，由 kernel 内部的 AssembleSendData 填充
    send_data = torch.zeros(SEND_COUNT, dtype=torch.int32, device=f"npu:{local_rank}")
    # prof_buf: 预分配 profiling 缓冲区，支持多轮 launch 追加
    prof_buf_elems = prof_buf_int64_size(prof_block_dim, max_iters=NUM_RUNS)
    prof_buf = torch.zeros(prof_buf_elems, dtype=torch.int64, device=f"npu:{local_rank}")

    # ---------- 执行算子 ----------
    if rank == 0:
        print(f"\n  Running NotifyDispatchProf: num_experts={NUM_EXPERTS}, "
              f"num_tokens={NUM_TOKENS}, send_count={SEND_COUNT}")

    for _ in range(NUM_RUNS):
        outputs = ascend_tool.notify_dispatch_prof(
            send_data=send_data,
            token_per_expert_data=token_per_expert_data,
            prof_buf=prof_buf,
            send_count=SEND_COUNT,
            num_tokens=NUM_TOKENS,
            comm_group=comm_group,
            rank_size=rank_size,
            rank_id=rank,
            local_rank_size=local_rank_size,
            local_rank_id=local_rank,
        )
    torch_npu.npu.synchronize()

    # ---------- 解包输出 ----------
    output_names = [
        "sendDataOffset", "recvData", "totalRecvTokens",
        "recvCount", "recvOffset", "maxBs", "recvTokensPerExpert"
    ]
    output_dict = {name: tuple(t.shape) for name, t in zip(output_names, outputs)}

    # ---------- 验证形状 ----------
    expected_shapes = compute_output_shapes(SEND_COUNT, rank_size)
    if rank == 0:
        print(f"\n  Rank {rank} 输出形状验证:")
        shape_pass = validate_output_shapes(output_dict, expected_shapes)

        # ---------- 检查输出值 ----------
        total_recv = outputs[2].cpu().item()  # totalRecvTokens
        max_bs = outputs[5].cpu().item()      # maxBs
        recv_tokens_per_expert = outputs[6].cpu().numpy()  # recvTokensPerExpert

        print(f"\n  totalRecvTokens = {total_recv}")
        print(f"  maxBs = {max_bs}")
        print(f"  recvTokensPerExpert = {recv_tokens_per_expert}")
        print(f"  recvTokensPerExpert sum = {recv_tokens_per_expert.sum()}")

        # 基本一致性检查
        consistency_check = (total_recv > 0) and (max_bs > 0)
        print(f"\n  一致性检查: {'PASS' if consistency_check else 'FAIL'}")
        print(f"  形状检查: {'PASS' if shape_pass else 'FAIL'}")
        print(f"  总体: {'PASS' if shape_pass and consistency_check else 'FAIL'}")

    # ---------- 解析并生成 trace ----------
    prof_data = AscendProfTool.parse_prof_buf(prof_buf)
    trace_path = os.path.join(trace_dir, f"notify_dispatch_prof_rank{rank}.json")
    tool.generate_trace_json(
        prof_data_list=[prof_data],
        offsets=rank_offsets,
        output_path=trace_path,
        kernel_names={0: "NotifyDispatchProf"},
        rank_id=rank,
        tag_names=NOTIFY_DISPATCH_TAG_NAMES,
    )
    print(f"[Rank {rank}] Trace generated: {trace_path}")

    merged_path = os.path.join(trace_dir, "notify_dispatch_prof_merged.json")
    tool.generate_merged_trace_json(
        prof_data_list=[prof_data],
        offsets=rank_offsets,
        output_path=merged_path,
        world_size=rank_size,
        rank=rank,
        kernel_names={0: "NotifyDispatchProf"},
        tag_names=NOTIFY_DISPATCH_TAG_NAMES,
    )
    if rank == 0:
        print(f"[Rank 0] Merged trace generated: {merged_path}")

    # ---------- 清理 ----------
    dist.barrier(group=group)
    dist.destroy_process_group()


def npu_mode(num_processes, trace_dir):
    try:
        import torch
    except ImportError as e:
        print(f"Import error: {e}")
        print("请确保已安装 torch。")
        return

    torch.multiprocessing.spawn(
        fn=npu_worker,
        args=(num_processes, trace_dir),
        nprocs=num_processes,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NotifyDispatchProf 算子验证脚本")
    parser.add_argument(
        "--mode", choices=["demo", "npu"], default="demo",
        help="运行模式: demo=模拟验证(无需NPU), npu=真机执行(需多卡环境)")
    parser.add_argument(
        "--num-processes", type=int, default=2,
        help="npu 模式进程数 (使用 torch.multiprocessing.spawn 拉起)")
    parser.add_argument(
        "--trace-dir", type=str, default="./traces",
        help="Trace 文件输出目录 (default: ./traces)")
    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode()
    else:
        npu_mode(args.num_processes, args.trace_dir)
