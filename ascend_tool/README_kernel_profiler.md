# Ascend Kernel Profiler — 完整使用手册

## 目录

- [1. 概述](#1-概述)
- [2. 系统架构](#2-系统架构)
- [3. 组件接口说明](#3-组件接口说明)
  - [3.1 Kernel 侧打点工具 — kernel_tool.h](#31-kernel-侧打点工具--kernel_toolh)
  - [3.2 时钟同步算子 — ProfCoreClockSync](#32-时钟同步算子--profcoreclocksynch)
  - [3.3 Python 工具类 — AscendProfTool](#33-python-工具类--ascendproftool)
  - [3.4 PyTorch Pybind 接口](#34-pytorch-pybind-接口)
- [4. 使用示例](#4-使用示例)
  - [4.1 Kernel 侧打点（基础）](#41-kernel-侧打点基础)
  - [4.2 Kernel 侧打点（多轮迭代）](#42-kernel-侧打点多轮迭代)
  - [4.3 Python 单卡完整流程](#43-python-单卡完整流程)
  - [4.4 Python 多卡完整流程](#44-python-多卡完整流程)
  - [4.5 VecAddProf 示例算子](#45-vecaddprof-示例算子)
- [5. 底层实现原理](#5-底层实现原理)
  - [5.1 Kernel 打点机制](#51-kernel-打点机制)
  - [5.2 GM 内存布局（多轮迭代）](#52-gm-内存布局多轮迭代)
  - [5.3 Cacheline 隔离保证](#53-cacheline-隔离保证)
  - [5.4 时钟同步原理](#54-时钟同步原理)
  - [5.5 多卡全局时钟对齐原理](#55-多卡全局时钟对齐原理)
  - [5.6 Trace Event JSON 生成](#56-trace-event-json-生成)
- [6. GM Buffer 大小计算](#6-gm-buffer-大小计算)
- [7. Cursor Skill：给任意算子使能 Prof 打点](#7-cursor-skill给任意算子使能-prof-打点)

---

## 1. 概述

本工具集提供 Ascend AI Core 上轻量级 Kernel 级别性能分析能力，支持单卡多核与多卡多核场景。整个系统由以下组件协同工作：

| 组件 | 位置 | 职责 |
|------|------|------|
| `kernel_tool.h` | Kernel 侧 (AI Core) | 打点记录 (tag, timestamp)，flush 到 GM |
| `prof_core_clock_sync.h` | Kernel 侧 (AI Core) | 单卡多核时钟同步 |
| `AscendProfTool` | Host 侧 (Python) | 时钟校准、buffer 解析、Trace JSON 生成 |
| Pybind 接口 | Host 侧 (C++/Python) | PyTorch 算子调用桥接 |

核心特性：
- 基于 `AscendC::GetSystemCycle()` 获取硬件时钟（50 cycles = 1 μs）
- 通过 `__BLOCK_LOCAL__` 实现每核独立的本地缓存，避免 cacheline false sharing
- 支持多轮迭代（多次 launch 共享同一 buffer），自动管理迭代计数
- 多核/多卡时钟偏移校准，丢弃 warmup 取中位数消除抖动
- 生成 Chrome Trace Event Format JSON，可直接用 `chrome://tracing` 或 Perfetto 查看
- 通过宏开关 `ASCEND_PROFILE_ENABLE` 实现零开销关闭

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           用户 Kernel                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │ PROF_INIT()  │───▶│PROF_RECORD() │───▶│ PROF_TO_GM() │                │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                │
│         kernel_tool.h                            │                       │
├──────────────────────────────────────────────────┼───────────────────────┤
│                       GM Buffer                  ▼                       │
│  ┌──────────┬───────────────┬─────────────────────────────────┐          │
│  │ Global   │  Core Headers │  Iter0 Data  │  Iter1 Data  │...│          │
│  │ Header   │  (per-core)   │  (per-core)  │  (per-core)  │   │          │
│  └──────────┴───────────────┴──────────────┴──────────────┴───┘          │
├──────────────────────────────────────────────────┬───────────────────────┤
│                    Host 侧                       │                       │
│  ┌────────────────────┐   ┌──────────────────────▼─────────────────┐     │
│  │ProfCoreClockSync   │   │ AscendProfTool                        │     │
│  │ (时钟校准算子)      │──▶│  .calibrate_core_clocks()             │     │
│  └────────────────────┘   │  .calibrate_rank_clocks()             │     │
│                           │  .parse_prof_buf()                    │     │
│                           │  .generate_trace_json()               │     │
│                           │  .generate_merged_trace_json()        │     │
│                           └───────────────────┬───────────────────┘     │
│                                               ▼                         │
│                                    trace.json (Perfetto / Chrome)       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 组件接口说明

### 3.1 Kernel 侧打点工具 — `kernel_tool.h`

#### 宏接口

| 宏 | 签名 | 功能 |
|----|------|------|
| `ASCEND_PROFILE_ENABLE` | 编译宏 | 定义此宏启用打点；不定义则所有打点宏展开为空操作（零开销） |
| `PROF_MAX_SLOTS` | 编译宏 (默认 `64`) | 每核每迭代的最大打点数，可在 `#include` 前自定义 |
| `PROF_MAX_ITERS` | 编译宏 (默认 `10`) | 最大迭代轮数，可在 `#include` 前自定义 |
| `PROF_INIT(gt)` | `GlobalTensor<int64_t>& gt` | 初始化：从 GM 读取当前迭代计数，重置本地缓存，自动记录起始时间戳 (tag=0) |
| `PROF_RECORD_TIME(tag)` | `int64_t tag` | 记录一个 (tag, timestamp) 对到本核本地缓存 |
| `PROF_RECORD_TIME_SYNC(tag)` | `int64_t tag` | 先执行 `PipeBarrier<PIPE_ALL>()`，再记录 (tag, timestamp) |
| `PROF_RECORD_TIME_SYNC(tag, X)` | `int64_t tag`, `pipe_t X` | 先执行 `PipeBarrier<X>()`，再记录 (tag, timestamp)。`X` 可选 `PIPE_V` / `PIPE_M` / `PIPE_S` / `PIPE_MTE1` / `PIPE_MTE2` / `PIPE_MTE3` / `PIPE_ALL` 等 |
| `PROF_TO_GM(gt)` | `GlobalTensor<int64_t>& gt` | 自动追加 ITER_END (tag=99999)，将本核本迭代数据 flush 到 GM，迭代计数 +1 |
| `PROF_SLEEP_US(us)` | `int64_t us` | busy-wait 指定微秒数（用于在 trace 中插入可视间隔） |
| `PROF_SYNC_ALL()` | — | 全核 barrier（`SyncAll<true>()`），用于创建对齐标记点 |
| `PROF_GM_BUF_SIZE(coreNum, maxIters)` | `int, int` | 计算所需 GM buffer 大小（字节） |

#### 核心函数

| 函数 | 说明 |
|------|------|
| `ProfileInit(gt)` | 从 GM 读 `iterCount`（首次 launch 为 0），重置打点索引，记录 tag=0 时间戳 |
| `RecordTime(tag)` | 调用 `GetSystemCycle()` 获取当前 cycle 并与 tag 一起写入本地 `__BLOCK_LOCAL__` 数组 |
| `RecordTimeSync<PipeType>(tag)` | `PipeBarrier<PipeType>()` + `RecordTime(tag)`。模板参数默认 `PIPE_ALL` |
| `ProfileToGm(gt)` | 自动追加 ITER_END sentinel，写入 Global Header（仅 core 0），写入本迭代数据，更新 `iterCount` |
| `SleepUs(us)` | 基于 `GetSystemCycle()` 的 busy-wait，50 cycles = 1 μs |

#### 关键常量 (AscendProf 命名空间)

| 常量 | 值 | 含义 |
|------|---|------|
| `CACHELINE_BYTES` | 64 | 物理 cacheline 大小（字节） |
| `CACHELINE_INT64` | 8 | 一个 cacheline 容纳的 int64 数 |
| `GLOBAL_HEADER_SIZE` | 8 | Global Header 大小（int64 数，1 cacheline） |
| `CORE_HEADER_SIZE` | 8 | 每核 Header 大小（int64 数，1 cacheline） |
| `CORE_META_SIZE` | 8 | 每迭代每核 meta 区大小（int64 数，1 cacheline） |
| `CORE_DATA_SIZE` | 128 | 每迭代每核数据区大小（`MAX_SLOTS×2`） |
| `ITER_CORE_STRIDE` | 136 | 每迭代核间间距（int64 数，17 cachelines） |

---

### 3.2 时钟同步算子 — `prof_core_clock_sync.h`

单卡多核时钟同步算子，通过多轮 `SyncAll` barrier 后立即抓取时间戳实现。

#### 工作流程

```
对每一轮 round (共 syncRounds 轮):
    SyncAll<true>()         // Barrier 1: 全核到达
    SyncAll<true>()         // Barrier 2: 消除前轮 GM 写延迟
    cycle = GetSystemCycle()  // 双 barrier 后立即采样
    写入 GM[core_base + round]
```

#### GM 输出布局

```
Header (8 int64, core 0 写入):
  [0] blockNum         — 核数
  [1] syncRounds       — 同步轮数
  [2] coreStride       — 核间步长 (int64 数)
  [3] coreRegionStart  — 核数据区起始索引

Per-Core Region (at coreRegionStart + i × coreStride):
  [+0 .. +(syncRounds-1)] : 各轮时间戳 (cycle)
```

---

### 3.3 Python 工具类 — `AscendProfTool`

```python
from ascend_prof_tool import AscendProfTool
```

#### 构造函数

```python
AscendProfTool(block_dim=None, sync_rounds=16)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `block_dim` | `int` or `None` | AI Core 数量。`None` 时自动从 NPU 设备属性获取 |
| `sync_rounds` | `int` | 时钟同步轮数，多轮取中位数消除抖动（默认 16） |

#### 方法一览

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `calibrate_core_clocks()` | `Dict[int, float]` | 单卡多核时钟校准。返回 `{core_id: offset_cycles}` |
| `calibrate_rank_clocks(ep_world_size, ep_rank_id, group=None)` | `Dict[Tuple[int,int], float]` | 多卡多核时钟校准。返回 `{(rank, core): offset_cycles}`；`group` 为集合通信使用的通信组 |
| `parse_prof_buf(buf)` | `Dict` | 解析 kernel_tool.h 生成的 GM buffer（支持多轮迭代） |
| `generate_trace_json(prof_data_list, offsets, output_path, ...)` | — | 生成单 rank 的 Chrome Trace JSON |
| `generate_merged_trace_json(prof_data_list, offsets, output_path, ...)` | — | 通过 allgather 合并所有 rank 生成统一 Trace JSON（仅 rank 0 写入） |

#### `parse_prof_buf` 返回结构

```python
{
    "block_num": int,             # 核数
    "max_slots": int,             # 每核每迭代最大打点数
    "max_iters": int,             # 最大迭代轮数
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
                    "count": int,       # 本迭代本核记录条数
                    "records": [
                        {"tag": int, "cycle": int},
                        ...
                    ]
                }, ...
            ]
        }, ...
    ]
}
```

#### `generate_trace_json` 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `prof_data_list` | `List[Dict]` | 多个 kernel 的打点数据列表（`parse_prof_buf` 返回值） |
| `offsets` | `Dict` | 时钟偏移表。单卡: `{core_id: offset}`，多卡: `{(rank, core): offset}` |
| `output_path` | `str` | 输出 JSON 文件路径 |
| `kernel_names` | `Dict[int, str]` | kernel 索引到名称的映射 |
| `rank_id` | `int` | 本 rank ID |
| `tag_names` | `Dict[int, str]` | tag 到可读标签片段的映射，如 `{1: "DMA_Start"}`，匹配的 tag 替换为自定义名称 |
| `tag_name_fn` | `callable` | 签名 `fn(tag_start, tag_end) -> str`，自定义命名逻辑 |

---

### 3.4 PyTorch Pybind 接口

通过 `ascend_tool` Python 模块暴露以下函数：

```python
import ascend_tool

# 向量加 + 打点
z = ascend_tool.vec_add_prof(x, y, prof_buf)

# 多核时钟同步
sync_timestamps = ascend_tool.prof_core_clock_sync(sync_buf)
```

| 函数 | 参数 | 返回值 |
|------|------|--------|
| `vec_add_prof(x, y, prof_buf)` | `x, y`: 输入 Tensor; `prof_buf`: int64 GM buffer | `z`: 输出 Tensor |
| `prof_core_clock_sync(sync_buf)` | `sync_buf`: int64 GM buffer | `sync_timestamps`: 时钟同步结果 Tensor |

---

## 4. 使用示例

### 4.1 Kernel 侧打点（基础）

```cpp
#define ASCEND_PROFILE_ENABLE      // 启用打点（注释此行则零开销关闭）
#include "kernel_tool.h"

extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, GM_ADDR profBuf, ...) {
    AscendC::GlobalTensor<int64_t> profGm;
    profGm.SetGlobalBuffer((__gm__ int64_t*)profBuf);

    PROF_INIT(profGm);             // 自动记录起始时间戳 (tag=0)

    // ... 业务逻辑阶段 1 (含 Vector 流水线操作) ...
    PROF_RECORD_TIME_SYNC(1, PIPE_V);  // 等待 Vector 流水线完成后记录

    // ... 业务逻辑阶段 2 ...
    PROF_RECORD_TIME_SYNC(2);          // 等待所有流水线完成后记录 (默认 PIPE_ALL)

    // ... 业务逻辑阶段 3 (纯搬运) ...
    PROF_RECORD_TIME(3);               // 不需要同步，直接记录

    PROF_TO_GM(profGm);            // 自动追加 ITER_END(99999)，flush 到 GM
}
```

Host 侧分配 buffer：

```cpp
int64_t bufSize = PROF_GM_BUF_SIZE(coreNum, 1);  // 单迭代，返回字节数
// 分配 bufSize 字节的零初始化 GM buffer
```

### 4.2 Kernel 侧打点（多轮迭代）

多次 launch 共享同一 buffer，每次 launch 为一次迭代：

```cpp
#define ASCEND_PROFILE_ENABLE
#define PROF_MAX_ITERS 4           // 最多记录 4 轮迭代
#include "kernel_tool.h"

extern "C" __global__ __aicore__ void my_kernel(GM_ADDR profBuf) {
    AscendC::GlobalTensor<int64_t> profGm;
    profGm.SetGlobalBuffer((__gm__ int64_t*)profBuf);

    // Launch 0: PROF_INIT 从 GM 读到 iterCount=0
    PROF_INIT(profGm);
    // ... work ...
    PROF_RECORD_TIME(1);
    PROF_TO_GM(profGm);      // flush iter 0, iterCount → 1

    // Launch 1: PROF_INIT 从 GM 读到 iterCount=1
    // ...以此类推
}
```

Host 侧：

```python
# Buffer 必须零初始化，且在多次 launch 之间不要清零
buf_size = PROF_GM_BUF_SIZE(core_num, max_iters)
prof_buf = torch.zeros(buf_size // 8, dtype=torch.int64, device="npu")

# 多次 launch 同一 kernel，共享 prof_buf
for _ in range(max_iters):
    my_kernel(prof_buf, ...)

torch.npu.synchronize()
prof_data = AscendProfTool.parse_prof_buf(prof_buf)
```

### 4.3 Python 单卡完整流程

```python
import torch
import torch_npu
from ascend_prof_tool import AscendProfTool

tool = AscendProfTool(sync_rounds=16)

# 步骤 1: 时钟校准（获取各核相对 core 0 的偏移）
offsets = tool.calibrate_core_clocks()
# offsets = {0: 0.0, 1: -2.5, 2: 1.0, ...}

# 步骤 2: 执行带打点的算子
import ascend_tool

N = 2048
x = torch.randn(N, dtype=torch.float16, device="npu")
y = torch.randn(N, dtype=torch.float16, device="npu")

# 分配 profiling buffer
buf_size = 8 + 8 * 8 + 1 * 8 * 136  # global_header + core_headers + data
prof_buf = torch.zeros(buf_size, dtype=torch.int64, device="npu")

z = ascend_tool.vec_add_prof(x, y, prof_buf)
torch.npu.synchronize()

# 步骤 3: 解析打点 buffer
prof_data = AscendProfTool.parse_prof_buf(prof_buf)

# 步骤 4: 生成 Trace JSON
tool.generate_trace_json(
    [prof_data], offsets, "trace_single.json",
    kernel_names={0: "VecAddProf"},
    tag_names={
        0: "Init", 999: "AllDone",
        100: "CopyIn[0]", 200: "Compute[0]", 300: "CopyOut[0]",
    },
)
# 用 chrome://tracing 或 Perfetto 打开 trace_single.json 即可查看
```

### 4.4 Python 多卡完整流程

```python
import torch
import torch.distributed as dist
import torch_npu
from ascend_prof_tool import AscendProfTool

# 假设已初始化分布式环境
ep_world_size = dist.get_world_size()
ep_rank_id = dist.get_rank()
ep_group = dist.new_group(list(range(ep_world_size)))

tool = AscendProfTool(sync_rounds=16)

# 步骤 1: 多卡时钟校准
# 内部流程: HCCL barrier → ProfCoreClockSync → HCCL allgather
offsets = tool.calibrate_rank_clocks(
    ep_world_size, ep_rank_id, group=ep_group
)
# offsets = {(0, 0): 0.0, (0, 1): -2.5, (1, 0): 150.0, ...}

# 步骤 2: 各 rank 独立执行带打点的算子并解析
prof_data = AscendProfTool.parse_prof_buf(prof_buf)

# 步骤 3a: 每个 rank 独立生成 trace
tool.generate_trace_json(
    [prof_data], offsets, f"trace_rank{ep_rank_id}.json",
    kernel_names={0: "MyKernel"}, rank_id=ep_rank_id,
)

# 步骤 3b: 或合并所有 rank 到一份 trace（需所有 rank 调用，仅 rank 0 写入）
tool.generate_merged_trace_json(
    [prof_data], offsets, "trace_merged.json",
    world_size=ep_world_size, rank=ep_rank_id,
    kernel_names={0: "MyKernel"},
)
```

### 4.5 VecAddProf 示例算子

`vec_add_prof` 是一个带打点插桩的向量加算子，演示了完整的打点用法：

```cpp
// vec_add_prof/op_kernel/vec_add_prof.cpp (简化)
#define ASCEND_PROFILE_ENABLE
#include "kernel_tool.h"

void Process() {
    PROF_INIT(profGm);

    for (int32_t i = 0; i < tileNum; i++) {
        PROF_SYNC_ALL();            // 全核对齐
        PROF_RECORD_TIME(10 + i);   // 同步标记
        PROF_SLEEP_US(10);          // 可视间隔
        PROF_RECORD_TIME(20 + i);   // 间隔结束

        CopyIn(i);                  // 内含 PROF_RECORD_TIME(100 + i)
        Compute(i);                 // 内含 PROF_RECORD_TIME(200 + i)
        CopyOut(i);                 // 内含 PROF_RECORD_TIME(300 + i)
    }

    PROF_SYNC_ALL();
    PROF_RECORD_TIME(998);          // 最终同步点
    PROF_RECORD_TIME(999);          // 全部完成
    PROF_TO_GM(profGm);
}
```

Tag 编码约定：

| Tag 范围 | 含义 |
|----------|------|
| 0 | Init（`PROF_INIT` 自动记录） |
| 10 + tile | 同步标记点 |
| 20 + tile | sleep 间隔结束 |
| 100 + tile | CopyIn 完成 |
| 200 + tile | Compute 完成 |
| 300 + tile | CopyOut 完成 |
| 998 | 最终同步点 |
| 999 | 全部完成 |
| 99999 | ITER_END（`PROF_TO_GM` 自动追加） |

---

## 5. 底层实现原理

### 5.1 Kernel 打点机制

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Core N (核内视角)                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  __BLOCK_LOCAL__ 存储 (每核独立)                        │    │
│  │  g_profileData[128]  — int64 数组，(tag, cycle) 交替存  │    │
│  │  g_profileDataIdx    — 当前写入位置 (0-based slot 索引)  │    │
│  │  g_profileIterCount  — 当前迭代轮次 (从 GM 读取)         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  PROF_INIT(gt):                                                 │
│    1. 从 GM 核 Header 读取 iterCount（首次为 0）                 │
│    2. 重置 g_profileDataIdx = 0                                 │
│    3. 调用 GetSystemCycle()，写入 {tag=0, cycle} 到槽位 0        │
│    4. g_profileDataIdx = 1                                      │
│                                                                 │
│  PROF_RECORD_TIME(tag):                                         │
│    1. 检查 idx < PROF_MAX_SLOTS                                 │
│    2. 调用 GetSystemCycle() 获取当前 cycle                       │
│    3. g_profileData[idx*2] = tag, [idx*2+1] = cycle             │
│    4. idx++                                                     │
│                                                                 │
│  PROF_RECORD_TIME_SYNC(tag [, PipeType]):                       │
│    1. PipeBarrier<PipeType>()  (默认 PIPE_ALL)                   │
│    2. 同 PROF_RECORD_TIME 的步骤 1-4                             │
│                                                                 │
│  PROF_TO_GM(gt):                                                │
│    1. 自动调用 RecordTime(99999) 追加 ITER_END sentinel         │
│    2. Core 0: 写入 Global Header [0..7]                         │
│    3. 计算本迭代本核写入位置:                                     │
│       iterBase = dataRgnStart + iter*blockNum*136 + blockIdx*136│
│    4. 写入 meta (recordCount) 和全部 data 到 GM                  │
│    5. 更新核 Header: iterCount++                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 GM 内存布局（多轮迭代）

#### 整体结构

```
  Global Header ｜ 各核 Header ｜ Iter0 全部核数据 ｜ Iter1 全部核数据 ｜...
```

#### 详细布局图（blockNum=2, maxIters=2）

```
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ [0]│ [1]│ [2]│ [3]│ [4]│ [5]│ [6]│ [7]│  Global Header (CL #0)
  │blk │max │max │core│core│iter│core│data│
  │Num │Slot│Iter│Hdr │Meta│Core│Hdr │Rgn │
  │    │    │    │Size│Size│Strd│Strt│Strt│
  └────┴────┴────┴────┴────┴────┴────┴────┘

  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ [8]│ [9]│[10]│[11]│[12]│[13]│[14]│[15]│  Core 0 Header (CL #1)
  │iter│rsv │rsv │rsv │rsv │rsv │rsv │rsv │
  │Cnt │    │    │    │    │    │    │    │
  └────┴────┴────┴────┴────┴────┴────┴────┘
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │[16]│[17]│[18]│[19]│[20]│[21]│[22]│[23]│  Core 1 Header (CL #2)
  │iter│rsv │rsv │rsv │rsv │rsv │rsv │rsv │
  │Cnt │    │    │    │    │    │    │    │
  └────┴────┴────┴────┴────┴────┴────┴────┘

  ─── Iter 0 Data ───────────────────────────
  ┌────┬────┬───── ... ─────┬────┐
  │[24]│[25]│  ...          │[31]│  Iter0/Core0 Meta (CL #3)
  │cnt │rsv │               │rsv │
  ├────┼────┼───── ... ─────┼────┤
  │[32]│[33]│  tag/ts pairs │    │  Iter0/Core0 Data (CL #4~19)
  │tag0│ ts0│  ...          │ts63│
  ├────┼────┼───── ... ─────┼────┤
  │160 │    │  ...          │167 │  Iter0/Core1 Meta (CL #20)
  │cnt │rsv │               │rsv │
  ├────┼────┼───── ... ─────┼────┤
  │168 │169 │  tag/ts pairs │    │  Iter0/Core1 Data (CL #21~36)
  └────┴────┴───── ... ─────┴────┘

  ─── Iter 1 Data ───────────────────────────
  │296 │ ...   Iter1/Core0 Meta + Data       │
  │432 │ ...   Iter1/Core1 Meta + Data       │
```

#### Global Header 字段 (buf[0..7]，仅 Core 0 写入)

| 索引 | 字段 | 含义 |
|------|------|------|
| `buf[0]` | `blockNum` | 参与的核数 |
| `buf[1]` | `maxSlots` | 每核每迭代最大槽位数 (默认 64) |
| `buf[2]` | `maxIters` | 最大迭代轮数 |
| `buf[3]` | `coreHeaderSize` | 每核 header 大小 (int64 数) |
| `buf[4]` | `coreMetaSize` | 每迭代每核 meta 大小 |
| `buf[5]` | `iterCoreStride` | 每迭代每核步长 (136 int64) |
| `buf[6]` | `coreHeaderRegionStart` | 各核 header 起始索引 |
| `buf[7]` | `dataRegionStart` | 迭代数据区起始索引 |

#### 每核 Header (at `buf[6] + i × buf[3]`)

| 偏移 | 字段 | 含义 |
|------|------|------|
| +0 | `iterCount` | 本核已完成的迭代次数 |
| +1~7 | reserved | — |

#### 每迭代每核数据 (at `buf[7] + j × blockNum × buf[5] + i × buf[5]`)

| 偏移 | 内容 | 说明 |
|------|------|------|
| +0 | `recordCount` | 本迭代本核记录的事件数 |
| +1~7 | reserved | 补齐 1 cacheline |
| +8 + k×2 | `tag_k` | 第 k 条事件 ID |
| +8 + k×2 + 1 | `ts_k` | 第 k 条时间戳 (cycle) |

### 5.3 Cacheline 隔离保证

| 保证项 | 机制 |
|--------|------|
| Global Header 不与任何核冲突 | 占 1 个完整 cacheline，仅 core 0 写入 |
| 核间 Header 隔离 | 每核 Header 独占 1 个 cacheline (8 int64) |
| 核间迭代数据隔离 | `ITER_CORE_STRIDE = 136` 是 `CACHELINE_INT64(8)` 的整数倍 |
| GM base 64B 对齐 | 假设 GM 基地址已对齐，所有区域自动对齐到物理 cacheline |
| 不同核永远不共享任何物理 cacheline | 由上述布局保证 |

### 5.4 时钟同步原理

**问题**：Ascend AI Core 各核的硬件时钟存在微小偏移，不同核的 `GetSystemCycle()` 返回值不可直接比较。

**解决方案**（`ProfCoreClockSync`）：

```
Round r (共 syncRounds 轮, 默认 16):
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Core 0  │     │ Core 1  │     │ Core N  │
  │         │     │         │     │         │
  │SyncAll()│────▶│SyncAll()│────▶│SyncAll()│   Barrier 1
  │SyncAll()│────▶│SyncAll()│────▶│SyncAll()│   Barrier 2 (消除写延迟)
  │ ts=Cyc()│     │ ts=Cyc()│     │ ts=Cyc()│   同一时刻采样
  │ GM[r]=ts│     │ GM[r]=ts│     │ GM[r]=ts│   各自写入 GM
  └─────────┘     └─────────┘     └─────────┘
```

Host 侧计算偏移：
1. 丢弃前 4 轮 warmup（消除冷启动抖动）
2. 对剩余轮次计算 `diff[r] = ts_core_i[r] - ts_core_0[r]`
3. 取中位数作为最终偏移

### 5.5 多卡全局时钟对齐原理

不同 NPU 卡的时钟域完全独立，光靠 `SyncAll` 无法跨卡同步。

**三段式流程**：

```
  Rank 0                     Rank 1
  ────────                   ────────
  HCCL allreduce (barrier) ←────────→ HCCL allreduce (barrier)
       ↓                              ↓
  ProfCoreClockSync ──────── 独立 ──── ProfCoreClockSync
       ↓                              ↓
  HCCL allgather ←───────────────────→ HCCL allgather
       ↓
  Host 解析全部 rank 时间戳，以 (rank0, core0) 为参考连算全局偏移
```

**关键依赖链**（通过 tensor 建立 NPU stream 执行顺序）：
- `allreduce` 输出 → 写入 `sync_buf` → `ProfCoreClockSync` 输入
- `ProfCoreClockSync` 输出 → `allgather` 输入

这确保了 `allreduce → ProfCoreClockSync → allgather` 的严格执行序。

### 5.6 Trace Event JSON 生成

生成 [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU) 兼容 JSON。

**映射规则**：
- `pid` = rank_id（每个 rank 一个进程 lane）
- `tid` = core_id（每个 core 一个线程 lane）
- 每对相邻打点 `(tag_j, ts_j) → (tag_{j+1}, ts_{j+1})` 生成一个 Duration Event (`ph: "X"`)
- 时间戳从 cycles 转换为微秒：`ts_us = ts_cycles / 50`
- 时钟偏移在转换前扣除

**事件默认命名规则**：

每个 Duration Event 跨越一对相邻打点 `(tag_start, tag_end)`，其 `name` 格式为 `Tag_<start>-<end>`。

解析优先级：

1. `tag_names` 字典（用户显式传入，匹配到的 tag 替换为自定义名称，未匹配的保留默认标签）
2. `tag_name_fn` 回调函数（签名 `fn(tag_start, tag_end) -> str`）
3. 内置默认规则（`_default_tag_name`）：

**标签片段规则**：

| Tag 值 | 标签片段 | 说明 |
|--------|----------|------|
| `0` | `Init` | `PROF_INIT` 自动记录的起始时间戳 |
| `99999` | `IterEnd` | `PROF_TO_GM` 自动追加的迭代结束 sentinel |
| 其他任意值 `N` | `N` | 直接使用数字，如 tag=1 → `1` |

**命名示例**：

| 起始 Tag | 结束 Tag | 默认事件名 |
|----------|----------|------------|
| 0 | 1 | `Tag_Init-1` |
| 1 | 2 | `Tag_1-2` |
| 2 | 99999 | `Tag_2-IterEnd` |
| 0 | 99999 | `Tag_Init-IterEnd` |
| 100 | 200 | `Tag_100-200` |

示例：若某核记录了 `[0, 1, 2, 99999]` 四个 tag，则生成 3 个 Duration Event：

```
| Tag_Init-1 | Tag_1-2 | Tag_2-IterEnd |
t0          t1       t2              t3
```

用户可通过 `tag_names` 覆盖任意 tag 的标签片段：

```python
tool.generate_trace_json(
    [prof_data], offsets, "trace.json",
    tag_names={1: "DMA_Start", 2: "Compute_Done"},
    # 0→1 变为 Tag_Init-DMA_Start
    # 1→2 变为 Tag_DMA_Start-Compute_Done
    # 2→99999 变为 Tag_Compute_Done-IterEnd
)
```

**查看方式**：
1. Chrome 浏览器打开 `chrome://tracing`，拖入 JSON 文件
2. 或使用 [Perfetto UI](https://ui.perfetto.dev/) 打开

---

## 6. GM Buffer 大小计算

```
PROF_GM_BUF_SIZE(coreNum, maxIters)
  = (GLOBAL_HEADER_SIZE + coreNum × CORE_HEADER_SIZE + maxIters × coreNum × ITER_CORE_STRIDE) × 8 bytes
  = (8 + coreNum × 8 + maxIters × coreNum × 136) × 8 bytes
```

| 核数 | 1 迭代 (bytes) | 4 迭代 (bytes) | 10 迭代 (bytes) |
|------|---------------|----------------|-----------------|
| 1 | 1,216 | 4,480 | 11,008 |
| 2 | 2,304 | 8,960 | 22,016 |
| 4 | 4,480 | 17,920 | 44,032 |
| 8 | 8,832 | 35,840 | 88,064 |
| 16 | 17,536 | 71,680 | 176,128 |
| 32 | 34,944 | 143,360 | 352,256 |

---

## 7. Cursor Skill：给任意算子使能 Prof 打点

项目内新增了一个可复用技能（Skill），用于把“给算子加 `prof_buf` + kernel 打点 + pybind/aclnn 接线 + 测试生成 trace”的整套流程标准化：

- 技能路径：`.cursor/skills/enable-ascend-op-prof/SKILL.md`
- 技能名称：`enable-ascend-op-prof`

### 7.1 什么时候用

当你要做以下任一任务时，建议直接触发此 Skill：

- 给某个 Ascend 自定义算子新增 `prof_buf` 入参
- 参考 `VecAddProf` 给 kernel 增加阶段打点
- 同步修改 pybind / `aclnn*GetWorkspaceSize` 参数签名
- 在 torch 用例中输出 rank trace + merged trace
- 排查 `parse_prof_buf` 越界（如 buffer 大小与实际核数不一致）

### 7.2 如何在 Cursor 中触发

在 Agent 对话中直接下达类似指令即可：

```text
请使用 enable-ascend-op-prof skill，给 Xxx 算子加 prof_buf，并补齐 pybind 和 trace 用例。
```

或：

```text
参考 enable-ascend-op-prof skill，把当前算子改造成可生成 trace 的 Prof 算子。
```

### 7.3 Skill 覆盖的关键检查项

该 Skill 默认会按以下链路检查并改造：

1. `op_host + json`：新增 `prof_buf`，并保证 `DataType/Format/UnknownShapeFormat` 组合组数一致
2. `op_kernel`：接入 `prof_buf`，在 `Process()` 中加入 `PROF_INIT` / `PROF_RECORD_TIME` / `PROF_TO_GM`
3. `pybind`：声明、实现、`TORCH_LIBRARY` schema 三处签名一致
4. `aclnn`：`GetWorkspaceSize` 签名透传 `profBuf`
5. `torch` 用例：`calibrate_rank_clocks(..., group=ep_group) -> parse_prof_buf -> generate_trace_json -> generate_merged_trace_json`
6. 保护项：避免 `prof_buf` 按错误核数分配导致解析越界

### 7.4 推荐配套实践

- 在用例里优先按运行时核数分配 `prof_buf`，不要长期硬编码核数
- 统一维护 `tag_names`，提升 trace 可读性
- 每次改造后至少验证：
  - 算子能正常执行
  - `parse_prof_buf` 能解析
  - trace 文件可在 `chrome://tracing` 或 Perfetto 打开
