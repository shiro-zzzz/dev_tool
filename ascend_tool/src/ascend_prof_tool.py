#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ascend_prof_tool.py — AscendProfTool

封装 ProfCoreClockSync 算子，结合 HCCL allgather 提供:
  1. 单卡/多卡时钟对齐
  2. 解析 kernel_tool.h 生成的打点 buffer
  3. 对齐后生成 Trace Event Format JSON (Chrome tracing / Perfetto 兼容)

多卡方案: 先通过 HCCL barrier 同步所有 rank, 确保各卡从同一时刻开始;
再每张卡独立调用 ProfCoreClockSync 完成卡内多核同步; 最后通过 HCCL
allgather 将各卡同步时间戳收集到一起, 由 host 侧计算全局偏移.
HCCL 通讯算子与 ProfCoreClockSync 之间通过 tensor 依赖保证顺序执行.

时钟换算: 50 cycles = 1 μs

使用示例:
    tool = AscendProfTool(sync_rounds=16)  # block_dim 自动从设备属性获取

    # --- 单卡场景 ---
    offsets = tool.calibrate_core_clocks()
    prof_data = tool.parse_prof_buf(prof_buf_tensor)
    tool.generate_trace_json(prof_data, offsets, "trace_rank0.json",
                             kernel_names={0: "MyKernel"})

    # --- 多卡场景 ---
    offsets = tool.calibrate_rank_clocks(ep_world_size, ep_rank_id)
    prof_data = tool.parse_prof_buf(prof_buf_tensor)
    # 每个 rank 独立生成 trace:
    tool.generate_trace_json(prof_data, offsets, f"trace_rank{ep_rank_id}.json",
                             kernel_names={0: "DispatchKernel"},
                             rank_id=ep_rank_id)
    # 或合并所有 rank 到一份 trace (所有 rank 都需调用, 仅 rank 0 写入):
    tool.generate_merged_trace_json(
        [prof_data], offsets, "trace_merged.json",
        world_size=ep_world_size, rank=ep_rank_id,
        kernel_names={0: "DispatchKernel"})
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple

CYCLES_PER_US = 50  # 50 cycles = 1 μs
PROF_ITER_END_TAG = 99999  # Iteration-end sentinel tag (matches kernel_tool.h)

# Safe upper bound for AI Core count across Ascend platforms
_MAX_AICORE_NUM = 64


class AscendProfTool:
    """Ascend 多核/多卡 Profiling 时钟对齐与 Trace 生成工具"""

    # Number of initial sync rounds to discard (warm-up)
    WARMUP_ROUNDS = 4

    def __init__(self, block_dim: Optional[int] = None, sync_rounds: int = 16):
        """
        Parameters
        ----------
        block_dim : int or None
            参与同步的 AI Core 数量. None 表示由算子根据平台自动检测,
            buffer 按 _MAX_AICORE_NUM 上界分配.
        sync_rounds : int
            同步轮数, 多轮取均值消除抖动 (默认 16).
        """
        self.block_dim = block_dim
        self.sync_rounds = sync_rounds

    # ================================================================
    # 1. 时钟对齐
    # ================================================================

    def calibrate_core_clocks(self) -> Dict[int, float]:
        """
        单卡多核时钟校准.

        调用 ProfCoreClockSync 算子, 返回各核相对 core 0 的时钟偏移 (cycles).

        Returns
        -------
        offsets : dict[int, float]
            offsets[core_id] = 该核相对 core 0 的平均偏移 (cycles).
            offsets[0] == 0.0 (参考核).
        """
        import torch
        import torch_npu  # noqa: F401
        import ascend_tool

        buf_size = self._core_sync_buf_int64()
        sync_buf = torch.zeros(buf_size, dtype=torch.int64, device="npu")

        result = ascend_tool.prof_core_clock_sync(sync_buf)
        torch.npu.synchronize()

        result_np = result.cpu().numpy()
        return self._compute_core_offsets(result_np)

    def calibrate_rank_clocks(
        self,
        ep_world_size: int,
        ep_rank_id: int,
    ) -> Dict[Tuple[int, int], float]:
        """
        多卡多核时钟校准.

        先通过 HCCL barrier (allreduce) 同步所有 rank, 确保各卡从同一时刻
        开始; 再每张卡独立调用 ProfCoreClockSync 完成卡内多核同步; 最后通过
        HCCL allgather 收集所有 rank 的同步时间戳, 计算全局偏移.

        通过 tensor 依赖建立 NPU stream 上的执行顺序:
        barrier → ProfCoreClockSync → allgather.

        Parameters
        ----------
        ep_world_size : int
            EP 通讯域中 rank 总数.
        ep_rank_id : int
            本 rank ID.

        Returns
        -------
        offsets : dict[(rank, core), float]
            offsets[(rank_id, core_id)] = 相对 (rank 0, core 0) 的偏移 (cycles).
        """
        import torch
        import torch_npu  # noqa: F401
        import torch.distributed as dist
        import ascend_tool

        buf_size = self._core_sync_buf_int64()

        # Step 1: HCCL barrier — 同步所有 rank, 确保各卡从同一时刻开始
        # 使用 allreduce 作为 barrier, 其输出 tensor 将写入 sync_buf
        # 以建立与后续 ProfCoreClockSync 的数据依赖.
        barrier_tensor = torch.ones(1, dtype=torch.int64, device="npu")
        dist.all_reduce(barrier_tensor, op=dist.ReduceOp.SUM)

        # Step 2: 调用 ProfCoreClockSync 完成卡内多核同步
        # 通过 slice 赋值将 barrier 输出写入 sync_buf, 纯 device 端操作,
        # 避免标量索引触发隐式 D2H 同步. 建立 NPU stream 依赖:
        # allreduce 完成 → sync_buf 写入 → ProfCoreClockSync 启动
        sync_buf = torch.zeros(buf_size, dtype=torch.int64, device="npu")
        sync_buf[:1].copy_(barrier_tensor)
        result = ascend_tool.prof_core_clock_sync(sync_buf)

        # Step 3: 通过 HCCL allgather 收集所有 rank 的同步时间戳
        # allgather 与 prof_core_clock_sync 通过 result tensor 建立依赖.
        gathered = [torch.zeros_like(result) for _ in range(ep_world_size)]
        dist.all_gather(gathered, result)

        torch.npu.synchronize()

        # 解析每个 rank 的同步结果并计算全局偏移
        all_rank_timestamps = {}  # (rank, core) -> list[cycle]
        for rank_id, rank_result in enumerate(gathered):
            rank_np = rank_result.cpu().numpy()
            header, per_core = self._parse_sync_output(rank_np)
            for core_id, cycles_per_round in per_core.items():
                all_rank_timestamps[(rank_id, core_id)] = cycles_per_round

        # 以 (rank 0, core 0) 为参考, 丢弃 warmup 轮, 取中位数
        ref_key = (0, 0)
        ref_ts = all_rank_timestamps.get(ref_key, [0] * self.sync_rounds)
        warmup = self.WARMUP_ROUNDS

        offsets = {}
        for (rank_id, core_id), ts_list in all_rank_timestamps.items():
            n = min(len(ts_list), len(ref_ts))
            if n > warmup:
                diffs = [ts_list[r] - ref_ts[r] for r in range(warmup, n)]
                offsets[(rank_id, core_id)] = float(np.median(diffs))
            elif n > 0:
                diffs = [ts_list[r] - ref_ts[r] for r in range(n)]
                offsets[(rank_id, core_id)] = float(np.median(diffs))
            else:
                offsets[(rank_id, core_id)] = 0.0

        return offsets

    # ================================================================
    # 2. 解析 kernel_tool.h 打点 buffer
    # ================================================================

    @staticmethod
    def parse_prof_buf(buf) -> Dict:
        """
        解析 kernel_tool.h 生成的 GM profiling buffer (支持多轮迭代).

        Parameters
        ----------
        buf : torch.Tensor (int64, CPU or NPU) 或 np.ndarray (int64)
            从 NPU 拷贝回的 profiling buffer.

        Returns
        -------
        dict : 解析后的结构化数据, 格式:
            {
                "block_num": int,
                "max_slots": int,
                "max_iters": int,
                "core_header_size": int,
                "core_meta_size": int,
                "iter_core_stride": int,
                "core_header_region_start": int,
                "data_region_start": int,
                "iterations": [
                    {
                        "iter_id": int,
                        "cores": [
                            {
                                "core_id": int,
                                "count": int,
                                "records": [
                                    {"tag": int, "cycle": int}, ...
                                ]
                            }, ...
                        ]
                    }, ...
                ]
            }
        """
        if hasattr(buf, "cpu"):
            buf_np = buf.cpu().numpy()
        else:
            buf_np = np.asarray(buf, dtype=np.int64)

        block_num = int(buf_np[0])
        max_slots = int(buf_np[1])
        max_iters = int(buf_np[2])
        core_header_size = int(buf_np[3])
        core_meta_size = int(buf_np[4])
        iter_core_stride = int(buf_np[5])
        core_header_region_start = int(buf_np[6])
        data_region_start = int(buf_np[7])

        # Read per-core iteration counts from core headers
        core_iter_counts = {}
        for i in range(block_num):
            hdr_base = core_header_region_start + i * core_header_size
            core_iter_counts[i] = int(buf_np[hdr_base])

        actual_iters = max(core_iter_counts.values()) if core_iter_counts else 0

        result = {
            "block_num": block_num,
            "max_slots": max_slots,
            "max_iters": max_iters,
            "core_header_size": core_header_size,
            "core_meta_size": core_meta_size,
            "iter_core_stride": iter_core_stride,
            "core_header_region_start": core_header_region_start,
            "data_region_start": data_region_start,
            "iterations": [],
        }

        for j in range(actual_iters):
            iter_data = {"iter_id": j, "cores": []}
            for i in range(block_num):
                if j >= core_iter_counts[i]:
                    continue
                base = (data_region_start
                        + j * block_num * iter_core_stride
                        + i * iter_core_stride)
                cnt = int(buf_np[base])
                records = []
                for k in range(cnt):
                    tag = int(buf_np[base + core_meta_size + k * 2])
                    ts = int(buf_np[base + core_meta_size + k * 2 + 1])
                    records.append({"tag": tag, "cycle": ts})
                iter_data["cores"].append({
                    "core_id": i,
                    "count": cnt,
                    "records": records,
                })
            result["iterations"].append(iter_data)

        return result

    # ================================================================
    # 3. 生成 Trace Event Format JSON
    # ================================================================

    def generate_trace_json(
        self,
        prof_data_list: List[Dict],
        offsets: Dict,
        output_path: str,
        kernel_names: Optional[Dict[int, str]] = None,
        rank_id: int = 0,
        tag_names: Optional[Dict[int, str]] = None,
        tag_name_fn=None,
    ):
        """
        生成 Chrome Trace Event Format JSON 文件.

        每个 kernel (prof_data_list 的每个元素) 的每个迭代占用一个独立泳道 (tid).
        每个 AI Core 占用一个独立进程 (pid).

        Parameters
        ----------
        prof_data_list : list[dict]
            多个 kernel 的打点数据, 每个是 parse_prof_buf() 的返回值.
            如果只有一个 kernel, 传 [prof_data] 即可.
            每个 prof_data 可包含多轮迭代 ("iterations" 键).
        offsets : dict
            时钟偏移表. 单卡: {core_id: offset}, 多卡: {(rank, core): offset}.
        output_path : str
            输出 JSON 文件路径.
        kernel_names : dict[int, str], optional
            kernel 索引到名称的映射. 默认 "Kernel_0", "Kernel_1", ...
        rank_id : int
            本 rank ID (用于多卡场景的 offset 查找).
        tag_names : dict[int, str], optional
            tag 到可读标签片段的映射, 例如 {1: "DMA_Start", 2: "Compute_Done"}.
            匹配到的 tag 替换为自定义名称, 优先级高于 tag_name_fn 和内置规则.
        tag_name_fn : callable, optional
            签名 fn(tag_start: int, tag_end: int) -> str, 将一对相邻 tag 转换为可读名称.
            默认使用内置规则 (Tag_<start>-<end>).
        """
        if kernel_names is None:
            kernel_names = {}
        if tag_name_fn is None:
            tag_name_fn = self._make_tag_name_fn(tag_names)
        elif tag_names:
            base_fn = tag_name_fn
            tag_name_fn = lambda s, e: tag_names.get(s, base_fn(s, e))

        trace_events = []

        # Collect all core IDs across all kernels and iterations
        all_core_ids = set()
        for prof_data in prof_data_list:
            for iter_data in prof_data.get("iterations", []):
                for core_data in iter_data["cores"]:
                    all_core_ids.add(core_data["core_id"])

        # pid = rank_id, process_name = "Rank {rank_id}"
        trace_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": rank_id,
            "args": {"name": f"Rank {rank_id}"},
        })

        # tid = core_id, thread_name = "Core {core_id}"
        for core_id in sorted(all_core_ids):
            trace_events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": rank_id,
                "tid": core_id,
                "args": {"name": f"Core {core_id}"},
            })

        # Duration events: all iterations on the same lane (tid = core_id)
        for kernel_idx, prof_data in enumerate(prof_data_list):
            kname = kernel_names.get(kernel_idx, f"Kernel_{kernel_idx}")
            iterations = prof_data.get("iterations", [])

            for iter_idx, iter_data in enumerate(iterations):
                for core_data in iter_data["cores"]:
                    core_id = core_data["core_id"]
                    records = core_data["records"]

                    # 获取时钟偏移
                    if isinstance(next(iter(offsets.keys()), 0), tuple):
                        offset = offsets.get((rank_id, core_id), 0.0)
                    else:
                        offset = offsets.get(core_id, 0.0)

                    # 生成 duration events (相邻打点之间为一个 duration event)
                    for j in range(len(records) - 1):
                        tag = records[j]["tag"]
                        tag_next = records[j + 1]["tag"]
                        ts_begin = records[j]["cycle"] - offset
                        ts_end = records[j + 1]["cycle"] - offset

                        # 转换为微秒
                        ts_begin_us = ts_begin / CYCLES_PER_US
                        dur_us = (ts_end - ts_begin) / CYCLES_PER_US

                        phase_name = tag_name_fn(tag, tag_next)

                        trace_events.append({
                            "name": phase_name,
                            "cat": kname,
                            "ph": "X",  # Complete event
                            "ts": ts_begin_us,
                            "dur": dur_us,
                            "pid": rank_id,
                            "tid": core_id,
                            "args": {
                                "kernel": kname,
                                "tag": tag,
                                "cycle_begin": int(ts_begin),
                                "cycle_end": int(ts_end),
                                "iter": iter_idx,
                            },
                        })

        trace_obj = {"traceEvents": trace_events}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace_obj, f, indent=2, ensure_ascii=False)

        total_iters = sum(len(pd.get("iterations", [])) for pd in prof_data_list)
        print(f"Trace JSON written to {output_path} "
              f"({len(trace_events)} events, "
              f"{len(prof_data_list)} kernels, "
              f"{total_iters} iterations, "
              f"{len(all_core_ids)} cores)")

    # ================================================================
    # 4. 多 rank trace 合并 (全通讯域生成一份)
    # ================================================================

    def generate_merged_trace_json(
        self,
        prof_data_list: List[Dict],
        offsets: Dict[Tuple[int, int], float],
        output_path: str,
        world_size: int,
        rank: int,
        kernel_names: Optional[Dict[int, str]] = None,
        tag_names: Optional[Dict[int, str]] = None,
        tag_name_fn=None,
    ):
        """
        多 rank trace 合并: 通过 HCCL allgather 收集所有 rank 的 prof_data,
        由 rank 0 生成一份包含所有 rank 数据的 Trace Event Format JSON.

        所有 rank 都必须调用此函数 (allgather 是集合通信), 但只有 rank 0
        会写入 JSON 文件.

        Parameters
        ----------
        prof_data_list : list[dict]
            本 rank 的 kernel 打点数据列表 (parse_prof_buf 返回值).
        offsets : dict[(int, int), float]
            全局时钟偏移表, key 为 (rank_id, core_id).
        output_path : str
            输出 JSON 文件路径 (仅 rank 0 写入).
        world_size : int
            通讯域 rank 总数.
        rank : int
            本 rank ID.
        kernel_names : dict[int, str], optional
            kernel 索引到名称的映射.
        tag_names : dict[int, str], optional
            tag 到可读标签片段的映射.
        tag_name_fn : callable, optional
            签名 fn(tag_start: int, tag_end: int) -> str.
        """
        import pickle
        import torch
        import torch.distributed as dist

        if kernel_names is None:
            kernel_names = {}
        if tag_name_fn is None:
            tag_name_fn = self._make_tag_name_fn(tag_names)
        elif tag_names:
            base_fn = tag_name_fn
            tag_name_fn = lambda s, e: tag_names.get(s, base_fn(s, e))

        # 序列化本 rank 的 prof_data_list
        local_bytes = pickle.dumps(prof_data_list)
        local_tensor = torch.tensor(
            list(local_bytes), dtype=torch.uint8, device="npu"
        )
        local_size = torch.tensor(
            [len(local_bytes)], dtype=torch.int64, device="npu"
        )

        # allgather sizes → allgather padded data
        all_sizes = [torch.zeros(1, dtype=torch.int64, device="npu")
                     for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)

        padded = torch.zeros(max_size, dtype=torch.uint8, device="npu")
        padded[:len(local_bytes)] = local_tensor

        all_padded = [torch.zeros(max_size, dtype=torch.uint8, device="npu")
                      for _ in range(world_size)]
        dist.all_gather(all_padded, padded)

        if rank != 0:
            return

        # rank 0: 反序列化各 rank 数据, 生成合并 trace
        trace_events = []
        for r in range(world_size):
            r_size = int(all_sizes[r].item())
            r_bytes = bytes(all_padded[r][:r_size].cpu().tolist())
            r_prof_data_list = pickle.loads(r_bytes)

            # process metadata
            all_core_ids = set()
            for prof_data in r_prof_data_list:
                for iter_data in prof_data.get("iterations", []):
                    for core_data in iter_data["cores"]:
                        all_core_ids.add(core_data["core_id"])

            trace_events.append({
                "name": "process_name", "ph": "M",
                "pid": r, "args": {"name": f"Rank {r}"},
            })
            for cid in sorted(all_core_ids):
                trace_events.append({
                    "name": "thread_name", "ph": "M",
                    "pid": r, "tid": cid,
                    "args": {"name": f"Core {cid}"},
                })

            # duration events
            for kernel_idx, prof_data in enumerate(r_prof_data_list):
                kname = kernel_names.get(kernel_idx, f"Kernel_{kernel_idx}")
                for iter_idx, iter_data in enumerate(
                    prof_data.get("iterations", [])
                ):
                    for core_data in iter_data["cores"]:
                        cid = core_data["core_id"]
                        records = core_data["records"]
                        offset = offsets.get((r, cid), 0.0)

                        for j in range(len(records) - 1):
                            tag = records[j]["tag"]
                            tag_next = records[j + 1]["tag"]
                            ts_begin = records[j]["cycle"] - offset
                            ts_end = records[j + 1]["cycle"] - offset
                            ts_begin_us = ts_begin / CYCLES_PER_US
                            dur_us = (ts_end - ts_begin) / CYCLES_PER_US

                            trace_events.append({
                                "name": tag_name_fn(tag, tag_next),
                                "cat": kname,
                                "ph": "X",
                                "ts": ts_begin_us,
                                "dur": dur_us,
                                "pid": r,
                                "tid": cid,
                                "args": {
                                    "kernel": kname,
                                    "tag": tag,
                                    "cycle_begin": int(ts_begin),
                                    "cycle_end": int(ts_end),
                                    "iter": iter_idx,
                                },
                            })

        trace_obj = {"traceEvents": trace_events}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace_obj, f, indent=2, ensure_ascii=False)

        total_events = sum(1 for e in trace_events if e.get("ph") == "X")
        print(f"Merged trace written to {output_path} "
              f"({len(trace_events)} events, {total_events} duration, "
              f"{world_size} ranks)")

    # ================================================================
    # Internal helpers
    # ================================================================

    def _resolve_block_dim(self) -> int:
        """Resolve block_dim: 显式指定则直接返回, 否则从 NPU 设备属性获取."""
        if self.block_dim is not None:
            return self.block_dim
        import torch
        import torch_npu
        device_id = torch.npu.current_device()
        prop = torch.npu.get_device_properties(device_id)
        return prop.vector_core_num

    def _core_sync_buf_int64(self) -> int:
        """ProfCoreClockSync 所需的 GM buffer 大小 (int64 元素数)."""
        cl = 8  # CACHELINE_INT64
        header = cl
        core_stride = ((self.sync_rounds + cl - 1) // cl) * cl
        num_cores = self._resolve_block_dim()
        return header + (cl - 1) + num_cores * core_stride

    def _compute_core_offsets(self, buf_np: np.ndarray) -> Dict[int, float]:
        """从 ProfCoreClockSync 输出计算各核偏移 (丢弃 warmup, 取中位数)."""
        header, per_core = self._parse_sync_output(buf_np)
        ref_ts = per_core.get(0, [0] * self.sync_rounds)
        warmup = self.WARMUP_ROUNDS
        offsets = {}
        for core_id, ts_list in per_core.items():
            n = min(len(ts_list), len(ref_ts))
            if n > warmup:
                diffs = [ts_list[r] - ref_ts[r] for r in range(warmup, n)]
                offsets[core_id] = float(np.median(diffs))
            elif n > 0:
                diffs = [ts_list[r] - ref_ts[r] for r in range(n)]
                offsets[core_id] = float(np.median(diffs))
            else:
                offsets[core_id] = 0.0
        return offsets

    @staticmethod
    def _parse_sync_output(buf_np: np.ndarray) -> Tuple[Dict, Dict[int, List[int]]]:
        """
        解析 ProfCoreClockSync 的 GM 输出.

        Returns
        -------
        header : dict
            Header 字段.
        per_core : dict[int, list[int]]
            per_core[core_id] = [cycle_round_0, cycle_round_1, ...]
        """
        block_num = int(buf_np[0])
        sync_rounds = int(buf_np[1])
        core_stride = int(buf_np[2])
        core_region_start = int(buf_np[3])

        header = {
            "block_num": block_num,
            "sync_rounds": sync_rounds,
            "core_stride": core_stride,
            "core_region_start": core_region_start,
        }

        # 额外字段 (多卡场景)
        if len(buf_np) > 5:
            header["ep_world_size"] = int(buf_np[4])
            header["ep_rank_id"] = int(buf_np[5])

        per_core = {}
        for i in range(block_num):
            base = core_region_start + i * core_stride
            cycles = []
            for r in range(sync_rounds):
                cycles.append(int(buf_np[base + r]))
            per_core[i] = cycles

        return header, per_core

    @classmethod
    def _make_tag_name_fn(cls, tag_names: Optional[Dict[int, str]] = None):
        """构造 tag 名称解析函数, 用户映射优先于内置规则."""
        def fn(tag_start: int, tag_end: int) -> str:
            if tag_names:
                start_name = tag_names.get(tag_start)
                end_name = tag_names.get(tag_end)
                if start_name is not None and end_name is not None:
                    return f"Tag_{start_name}-{end_name}"
                if start_name is not None:
                    return f"Tag_{start_name}-{cls._tag_label(tag_end)}"
                if end_name is not None:
                    return f"Tag_{cls._tag_label(tag_start)}-{end_name}"
            return cls._default_tag_name(tag_start, tag_end)
        return fn

    @staticmethod
    def _tag_label(tag: int) -> str:
        """将单个 tag 转换为标签片段."""
        if tag == 0:
            return "Init"
        if tag == PROF_ITER_END_TAG:
            return "IterEnd"
        return str(tag)

    @classmethod
    def _default_tag_name(cls, tag_start: int, tag_end: int) -> str:
        """将一对相邻 tag 转换为可读的 Duration Event 名称."""
        return f"Tag_{cls._tag_label(tag_start)}-{cls._tag_label(tag_end)}"
