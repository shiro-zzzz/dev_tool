#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vec_add_prof.py — VecAddProf 算子打点验证脚本

使用方法:
  1. 先构建算子: compile_ascend_proj.sh (确保 VecAddProf.json 被 msopgen 处理)
  2. 安装生成的 .run 包
  3. 构建 pybind wheel 并安装
  4. 运行: python test_vec_add_prof.py

约束:
  - totalLength 必须是 BLOCK_DIM(8) × TILE_LENGTH(256) = 2048 的整数倍
"""

import numpy as np

PHASE_NAMES = {
    0: "Init",
    999: "AllDone",
}


def get_phase_name(tag):
    """将 tag 解码为可读的阶段名"""
    if tag in PHASE_NAMES:
        return PHASE_NAMES[tag]
    if 100 <= tag < 200:
        return f"CopyIn[tile={tag - 100}]"
    if 200 <= tag < 300:
        return f"Compute[tile={tag - 200}]"
    if 300 <= tag < 400:
        return f"CopyOut[tile={tag - 300}]"
    return f"Unknown[{tag}]"


def parse_prof_buf(buf_np):
    """
    解析从 VecAddProf 内核写入的 GM profiling buffer.

    Parameters
    ----------
    buf_np : np.ndarray, dtype=int64
        从 NPU 拷贝回 CPU 的 profiling buffer.

    Returns
    -------
    dict : 解析后的结构化数据
    """
    block_num         = int(buf_np[0])
    max_slots         = int(buf_np[1])
    core_stride       = int(buf_np[2])
    core_region_start = int(buf_np[3])
    core_meta_size    = int(buf_np[4])

    print("=" * 70)
    print("  Ascend Kernel Profiler — GM Buffer 解析结果")
    print("=" * 70)
    print(f"  blockNum        = {block_num}")
    print(f"  maxSlots        = {max_slots}")
    print(f"  coreStride      = {core_stride}")
    print(f"  coreRegionStart = {core_region_start}")
    print(f"  coreMetaSize    = {core_meta_size}")
    print("-" * 70)

    result = {
        "block_num": block_num,
        "max_slots": max_slots,
        "core_stride": core_stride,
        "core_region_start": core_region_start,
        "core_meta_size": core_meta_size,
        "cores": [],
    }

    for i in range(block_num):
        base = core_region_start + i * core_stride
        cnt  = int(buf_np[base])

        print(f"\n  Core {i}: {cnt} records")

        core_records = []
        init_ts = None

        for j in range(cnt):
            tag = int(buf_np[base + core_meta_size + j * 2])
            ts  = int(buf_np[base + core_meta_size + j * 2 + 1])

            if j == 0:
                init_ts = ts
            delta = ts - init_ts if init_ts is not None else 0
            phase = get_phase_name(tag)

            print(f"    [{j:3d}] tag={tag:4d}  phase={phase:<22s}  "
                  f"cycle={ts}  delta={delta}")

            core_records.append({
                "tag": tag, "phase": phase,
                "cycle": ts, "delta": delta,
            })

        result["cores"].append({"core_id": i, "count": cnt, "records": core_records})

    print("\n" + "=" * 70)

    # Summary: 每个核的总耗时
    print("\n  Summary (Init → AllDone):")
    for core_data in result["cores"]:
        records = core_data["records"]
        if len(records) >= 2:
            total = records[-1]["cycle"] - records[0]["cycle"]
            print(f"    Core {core_data['core_id']}: {total} cycles "
                  f"({len(records)} events)")

    print()
    return result


def test_vec_add_prof():
    """端到端测试: 运行 VecAddProf 算子并解析打点数据"""
    try:
        import torch
        import torch_npu
        import ascend_tool
    except ImportError as e:
        print(f"Import error: {e}")
        print("请确保已安装 torch, torch_npu, 以及 ascend_tool wheel 包.")
        print("\n以下以模拟数据演示解析逻辑:\n")
        demo_parse()
        return

    # ---------- 参数 ----------
    BLOCK_DIM    = 8
    TILE_LENGTH  = 256
    N            = BLOCK_DIM * TILE_LENGTH * 4   # 8192 elements, 4 tiles per core

    # ---------- 分配 ----------
    x = torch.randn(N, dtype=torch.float16, device="npu")
    y = torch.randn(N, dtype=torch.float16, device="npu")

    # profiling buffer: worst-case size = (15 + BLOCK_DIM * 136) int64 elements
    prof_size = 15 + BLOCK_DIM * 136
    prof_buf  = torch.zeros(prof_size, dtype=torch.int64, device="npu")

    # ---------- 执行 ----------
    z = ascend_tool.vec_add_prof(x, y, prof_buf)
    torch.npu.synchronize()

    # ---------- 验证正确性 ----------
    z_cpu = z.cpu().float()
    x_cpu = x.cpu().float()
    y_cpu = y.cpu().float()
    ref   = x_cpu + y_cpu
    max_err = (z_cpu - ref).abs().max().item()
    print(f"VecAdd correctness: max_error = {max_err:.6f}  "
          f"({'PASS' if max_err < 0.01 else 'FAIL'})")

    # ---------- 解析打点 ----------
    prof_np = prof_buf.cpu().numpy()
    parse_prof_buf(prof_np)


def demo_parse():
    """用模拟数据演示 parse_prof_buf，不依赖 NPU 硬件"""
    BLOCK_DIM  = 2   # 模拟 2 核
    TILE_NUM   = 3   # 每核 3 个 tile
    MAX_SLOTS  = 64
    CL         = 8
    META_SZ    = 8
    STRIDE     = 136
    HEADER_SZ  = 8
    REGION_START = HEADER_SZ  # 假设对齐

    buf_size = REGION_START + BLOCK_DIM * STRIDE
    buf = np.zeros(buf_size, dtype=np.int64)

    # Header
    buf[0] = BLOCK_DIM
    buf[1] = MAX_SLOTS
    buf[2] = STRIDE
    buf[3] = REGION_START
    buf[4] = META_SZ

    # 模拟每个核的数据
    base_cycle = 1000000
    for core in range(BLOCK_DIM):
        base = REGION_START + core * STRIDE
        cycle = base_cycle + core * 500

        # 事件列表: init, then per-tile CopyIn/Compute/CopyOut, then AllDone
        events = [(0, cycle)]
        for t in range(TILE_NUM):
            cycle += 100
            events.append((100 + t, cycle))
            cycle += 200
            events.append((200 + t, cycle))
            cycle += 50
            events.append((300 + t, cycle))
        cycle += 10
        events.append((999, cycle))

        buf[base] = len(events)  # recordCount
        for j, (tag, ts) in enumerate(events):
            buf[base + META_SZ + j * 2]     = tag
            buf[base + META_SZ + j * 2 + 1] = ts

    parse_prof_buf(buf)


if __name__ == "__main__":
    test_vec_add_prof()
