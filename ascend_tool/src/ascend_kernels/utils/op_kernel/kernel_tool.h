#ifndef KERNEL_TOOL_H
#define KERNEL_TOOL_H

#include <kernel_operator.h>

// ============================================================
// Ascend Kernel Profiler (Multi-Iteration)
// ============================================================
//
// Usage:
//   #define ASCEND_PROFILE_ENABLE   // define before including this header to enable
//   #define PROF_MAX_ITERS 4        // max iterations (default 10)
//   #include "kernel_tool.h"
//
//   // 一次 launch = 一次迭代, 多次 launch 共享同一 buffer 实现多迭代:
//   // Launch 0:
//   PROF_INIT(profGlobalTensor);    // 从 GM 读 iterCount (首次=0), record Init tag
//   // ... work ...
//   PROF_RECORD_TIME(1);
//   PROF_TO_GM(profGlobalTensor);   // auto-record ITER_END(99999), flush iter 0, iterCount → 1
//   // Launch 1:
//   PROF_INIT(profGlobalTensor);    // 从 GM 读 iterCount = 1
//   // ... work ...
//   PROF_RECORD_TIME(1);
//   PROF_TO_GM(profGlobalTensor);   // flush iter 1, iterCount → 2
//
// GM Layout (int64_t units, 64-byte cacheline = 8 x int64_t):
//
// ┌──────────────────────────────────────────────────────────────────────────┐
// │  Cacheline = 64 Bytes = 8 × int64_t                                    │
// │  GLOBAL_HEADER_SIZE = 8   (1 cacheline)                                 │
// │  CORE_HEADER_SIZE   = 8   (1 cacheline per core)                        │
// │  CORE_META_SIZE     = 8   (1 cacheline, per-iter per-core meta)         │
// │  CORE_DATA_SIZE     = 128 (MAX_SLOTS×2, 16 cachelines)                  │
// │  ITER_CORE_STRIDE   = 136 (17 cachelines)                               │
// │                                                                         │
// │  GM 首地址默认 64B 对齐.                                                 │
// └──────────────────────────────────────────────────────────────────────────┘
//
// 总布局:
//   总 Header ｜ 各核 Header ｜ 所有核迭代0打点数据 ｜ 所有核迭代1打点数据 ｜...
//
// 举例: blockNum=2, maxIters=2
// ══════════════════════════════════════════════════════════════════════════
//
//  ┌────┬────┬────┬────┬────┬────┬────┬────┐
//  │ [0]│ [1]│ [2]│ [3]│ [4]│ [5]│ [6]│ [7]│  Global Header (CL #0)
//  │blk │max │max │core│core│iter│core│data│
//  │Num │Slot│Iter│Hdr │Meta│Core│Hdr │Rgn │
//  │    │    │    │Size│Size│Strd│Strt│Strt│
//  └────┴────┴────┴────┴────┴────┴────┴────┘
//
//  ┌────┬────┬────┬────┬────┬────┬────┬────┐
//  │ [8]│ [9]│[10]│[11]│[12]│[13]│[14]│[15]│  Core 0 Header (CL #1)
//  │iter│rsv │rsv │rsv │rsv │rsv │rsv │rsv │
//  │Cnt │    │    │    │    │    │    │    │
//  └────┴────┴────┴────┴────┴────┴────┴────┘
//  ┌────┬────┬────┬────┬────┬────┬────┬────┐
//  │[16]│[17]│[18]│[19]│[20]│[21]│[22]│[23]│  Core 1 Header (CL #2)
//  │iter│rsv │rsv │rsv │rsv │rsv │rsv │rsv │
//  │Cnt │    │    │    │    │    │    │    │
//  └────┴────┴────┴────┴────┴────┴────┴────┘
//
//  ── Iter 0 Data ──────────────────────────
//  ┌────┬────┬────┬────┬────┬────┬────┬────┐
//  │[24]│[25]│    │    │    │    │    │[31]│  Iter0/Core0 Meta (CL #3)
//  │cnt │rsv │    │    │    │    │    │rsv │
//  ├────┼────┼────┼────┼────┼────┼────┼────┤
//  │[32]│[33]│[34]│[35]│    │    │    │    │  Iter0/Core0 Data (CL #4~19)
//  │tag0│ ts0│tag1│ ts1│ .. │ .. │ .. │ .. │
//  │ ...│    │    │    │    │    │t63 │ts63│
//  ├────┼────┼────┼────┼────┼────┼────┼────┤
//  │160 │    │    │    │    │    │    │167 │  Iter0/Core1 Meta (CL #20)
//  │cnt │rsv │    │    │    │    │    │rsv │
//  ├────┼────┼────┼────┼────┼────┼────┼────┤
//  │168 │169 │    │    │    │    │    │    │  Iter0/Core1 Data (CL #21~36)
//  │tag0│ ts0│ .. │ .. │ .. │ .. │t63 │ts63│
//  └────┴────┴────┴────┴────┴────┴────┴────┘
//
//  ── Iter 1 Data ──────────────────────────
//  ┌────┬────┬────┬────┬────┬────┬────┬────┐
//  │296 │    │    │    │    │    │    │303 │  Iter1/Core0 Meta
//  │cnt │rsv │ .. │ .. │ .. │ .. │ .. │rsv │
//  ├────┼────┼────┼────┼────┼────┼────┼────┤
//  │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │  Iter1/Core0 Data
//  └────┴────┴────┴────┴────┴────┴────┴────┘
//  │ ...  Iter1/Core1 ...                   │
//
// ══════════════════════════════════════════════════════════════════════════
//
// Global Header Fields (index 0..7, written by core 0 only):
//
//  ┌──────┬──────────────────────────────────────────────────┐
//  │Index │ Field                                            │
//  ├──────┼──────────────────────────────────────────────────┤
//  │  0   │ blockNum            — 参与的核数                  │
//  │  1   │ maxSlots            — 每核每迭代最大槽位数         │
//  │  2   │ maxIters            — 最大迭代轮数 (PROF_MAX_ITERS)│
//  │  3   │ coreHeaderSize      — 每核 header 大小 (int64 数)  │
//  │  4   │ coreMetaSize        — 每迭代每核 meta 大小         │
//  │  5   │ iterCoreStride      — 每迭代每核步长 (int64 数)    │
//  │  6   │ coreHeaderRegionStart — 各核 header 起始索引       │
//  │  7   │ dataRegionStart     — 迭代数据区起始索引           │
//  └──────┴──────────────────────────────────────────────────┘
//
// Per-Core Header (at coreHeaderRegionStart + i × coreHeaderSize):
//
//  ┌──────┬──────────────────────────────────────────────────┐
//  │Offset│ Field                                            │
//  ├──────┼──────────────────────────────────────────────────┤
//  │  +0  │ iterCount — 本核已完成的迭代次数                   │
//  │ +1~7 │ reserved                                         │
//  └──────┴──────────────────────────────────────────────────┘
//
// Per-Iteration Per-Core Data
//   (at dataRegionStart + j × blockNum × iterCoreStride + i × iterCoreStride):
//
//  ┌──────┬──────────────────────────────────────────────────┐
//  │Offset│ Field                                            │
//  ├──────┼──────────────────────────────────────────────────┤
//  │  +0  │ recordCount   — 本迭代本核实际记录的事件数         │
//  │ +1~7 │ reserved (补齐 1 cacheline)                       │
//  │  +8  │ tag_0  (事件 ID)                                  │
//  │  +9  │ ts_0   (时间戳 cycle)                             │
//  │ +10  │ tag_1                                             │
//  │ +11  │ ts_1                                              │
//  │ ...  │ ...                                               │
//  │+134  │ tag_63                                            │
//  │+135  │ ts_63                                             │
//  └──────┴──────────────────────────────────────────────────┘
//
// Summary Table:
//
//  ┌──────────────────────┬──────────────┬────────────┬──────────────────────────────────────────────┐
//  │ Region               │ Size(int64)  │ Cachelines │ Offset Formula                               │
//  ├──────────────────────┼──────────────┼────────────┼──────────────────────────────────────────────┤
//  │ Global Header        │      8       │     1      │ 0 (固定)                                     │
//  │ Core i Header        │      8       │     1      │ 8 + i×8                                      │
//  │ Iter j Core i Meta   │      8       │     1      │ dataRgnStart + j×N×136 + i×136               │
//  │ Iter j Core i Data   │    128       │    16      │ dataRgnStart + j×N×136 + i×136 + 8           │
//  │ Iter j Core i Total  │    136       │    17      │ (= ITER_CORE_STRIDE)                         │
//  ├──────────────────────┼──────────────┼────────────┼──────────────────────────────────────────────┤
//  │ Total (N核, M迭代)   │ 8+N×8+M×N×136│           │ dataRgnStart = 8+N×8                          │
//  └──────────────────────┴──────────────┴────────────┴──────────────────────────────────────────────┘
//
// Cacheline Isolation Guarantee:
//   - GM base address is assumed 64B aligned.
//   - Global header (8 int64) = 1 cacheline, written by core 0 only.
//   - Each core header (8 int64) = 1 cacheline, written by its own core only.
//   - ITER_CORE_STRIDE (136) is a multiple of CACHELINE_INT64 (8), so every
//     core's per-iteration region starts on a cacheline boundary.
//   - Different cores NEVER share any physical cacheline.
//
// Host parsing pseudocode:
//   blockNum             = buf[0]
//   maxSlots             = buf[1]
//   maxIters             = buf[2]
//   coreHeaderSize       = buf[3]
//   coreMetaSize         = buf[4]
//   iterCoreStride       = buf[5]
//   coreHeaderRegionStart= buf[6]
//   dataRegionStart      = buf[7]
//   for i in 0..blockNum-1:
//     iterCount = buf[coreHeaderRegionStart + i * coreHeaderSize]
//     for j in 0..iterCount-1:
//       base = dataRegionStart + j * blockNum * iterCoreStride + i * iterCoreStride
//       cnt  = buf[base]
//       for k in 0..cnt-1:
//         tag = buf[base + coreMetaSize + k*2]
//         ts  = buf[base + coreMetaSize + k*2 + 1]
//
// Required GM buffer size (bytes):
//   PROF_GM_BUF_SIZE(coreNum, maxIters)
// ============================================================

#ifndef PROF_MAX_SLOTS
#define PROF_MAX_SLOTS 64
#endif

#ifndef PROF_MAX_ITERS
#define PROF_MAX_ITERS 10
#endif

// Iteration-end sentinel tag (automatically recorded by PROF_TO_GM)
#define PROF_ITER_END_TAG 99999

namespace AscendProf {

constexpr int32_t CACHELINE_BYTES = 64;
constexpr int32_t CACHELINE_INT64 = CACHELINE_BYTES / static_cast<int32_t>(sizeof(int64_t)); // 8

__aicore__ constexpr int32_t AlignUp(int32_t x, int32_t align)
{
    return ((x + align - 1) / align) * align;
}

constexpr int32_t GLOBAL_HEADER_SIZE = CACHELINE_INT64;                                           // 8
constexpr int32_t CORE_HEADER_SIZE   = CACHELINE_INT64;                                           // 8
constexpr int32_t CORE_META_SIZE     = CACHELINE_INT64;                                           // 8
constexpr int32_t CORE_DATA_SIZE     = PROF_MAX_SLOTS * 2;                                        // 128
constexpr int32_t ITER_CORE_STRIDE   = AlignUp(CORE_META_SIZE + CORE_DATA_SIZE, CACHELINE_INT64); // 136

} // namespace AscendProf

// ------------------------------------------------------------
// Block-local profiling storage (each AI core owns its copy)
// ------------------------------------------------------------
__BLOCK_LOCAL__ __inline__ int64_t g_profileData[PROF_MAX_SLOTS * 2];
__BLOCK_LOCAL__ __inline__ int32_t g_profileDataIdx;
__BLOCK_LOCAL__ __inline__ int32_t g_profileIterCount;

// ------------------------------------------------------------
// ProfileInit  — read iterCount from GM, reset local state,
//                record the start timestamp (tag = 0).
//   On a zero-initialized buffer iterCount reads as 0 (first launch).
//   On subsequent launches iterCount reflects previous flushes.
// ------------------------------------------------------------
__aicore__ inline void ProfileInit(AscendC::GlobalTensor<int64_t>& gt)
{
    using namespace AscendProf;

    int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t coreHeaderBase = GLOBAL_HEADER_SIZE + blockIdx * CORE_HEADER_SIZE;

    // Read current iteration count from GM (0 if buffer was zero-initialized)
    g_profileIterCount = static_cast<int32_t>(gt.GetValue(coreHeaderBase));
    g_profileDataIdx = 0;

    int64_t cycle = static_cast<int64_t>(AscendC::GetSystemCycle());
    g_profileData[0] = 0;      // tag 0 = init
    g_profileData[1] = cycle;
    g_profileDataIdx = 1;
}

// ------------------------------------------------------------
// SleepUs  — busy-wait for the specified number of microseconds
//            (50 cycles = 1 μs on Ascend AI Core)
// ------------------------------------------------------------
constexpr int64_t CYCLES_PER_US = 50;

__aicore__ inline void SleepUs(int64_t us)
{
    int64_t start = static_cast<int64_t>(AscendC::GetSystemCycle());
    int64_t target = us * CYCLES_PER_US;
    while (static_cast<int64_t>(AscendC::GetSystemCycle()) - start < target) {
        // busy wait
    }
}

// ------------------------------------------------------------
// RecordTime  — record a (tag, timestamp) pair at the current slot
// ------------------------------------------------------------
__aicore__ inline void RecordTime(int64_t tag)
{
    int32_t idx = g_profileDataIdx;
    if (idx < PROF_MAX_SLOTS) {
        int64_t cycle = static_cast<int64_t>(AscendC::GetSystemCycle());
        g_profileData[idx * 2]     = tag;
        g_profileData[idx * 2 + 1] = cycle;
        g_profileDataIdx = idx + 1;
    }
}

// ------------------------------------------------------------
// RecordTimeSync — pipeline barrier + record a (tag, timestamp) pair
//   PipeBarrier<PipeType>() is called before recording.
//   Default PipeType = PIPE_ALL.
// ------------------------------------------------------------
template <pipe_t PipeType = PIPE_ALL>
__aicore__ inline void RecordTimeSync(int64_t tag)
{
    AscendC::PipeBarrier<PipeType>();
    RecordTime(tag);
}

// ------------------------------------------------------------
// ProfileToGm — flush the current iteration's profiling data to GM.
//   One launch = one iteration. Call exactly once per launch.
//
// Workflow:
//   1. Record PROF_ITER_END_TAG (99999) timestamp
//   2. Write global header (core 0 only)
//   3. Write per-iteration per-core data to GM
//   4. Update per-core header (iterCount++)
//
// GM base address is assumed 64B aligned.
// ------------------------------------------------------------
__aicore__ inline void ProfileToGm(AscendC::GlobalTensor<int64_t>& gt)
{
    using namespace AscendProf;

    // ---- Step 1: Record iteration-end sentinel ----
    RecordTime(PROF_ITER_END_TAG);

    int32_t blockIdx = static_cast<int32_t>(AscendC::GetBlockIdx());
    int32_t blockNum = static_cast<int32_t>(AscendC::GetBlockNum());
    int32_t cnt      = g_profileDataIdx;
    int32_t iter     = g_profileIterCount;

    int32_t coreHeaderRegionStart = GLOBAL_HEADER_SIZE;
    int32_t dataRegionStart       = GLOBAL_HEADER_SIZE + blockNum * CORE_HEADER_SIZE;

    // ---- Step 2: Global Header (indices 0..7, core 0 only) ----
    if (blockIdx == 0) {
        gt.SetValue(0, static_cast<int64_t>(blockNum));
        gt.SetValue(1, static_cast<int64_t>(PROF_MAX_SLOTS));
        gt.SetValue(2, static_cast<int64_t>(PROF_MAX_ITERS));
        gt.SetValue(3, static_cast<int64_t>(CORE_HEADER_SIZE));
        gt.SetValue(4, static_cast<int64_t>(CORE_META_SIZE));
        gt.SetValue(5, static_cast<int64_t>(ITER_CORE_STRIDE));
        gt.SetValue(6, static_cast<int64_t>(coreHeaderRegionStart));
        gt.SetValue(7, static_cast<int64_t>(dataRegionStart));
    }

    // ---- Step 3: Per-iteration per-core data ----
    int32_t iterBase = dataRegionStart
                     + iter * blockNum * ITER_CORE_STRIDE
                     + blockIdx * ITER_CORE_STRIDE;

    // Meta: record count
    gt.SetValue(iterBase, static_cast<int64_t>(cnt));

    // Data: (tag, timestamp) pairs
    int32_t dataBase = iterBase + CORE_META_SIZE;
    for (int32_t i = 0; i < cnt; i++) {
        gt.SetValue(dataBase + i * 2,     g_profileData[i * 2]);
        gt.SetValue(dataBase + i * 2 + 1, g_profileData[i * 2 + 1]);
    }

    // ---- Step 4: Update per-core header (iterCount) ----
    int32_t coreHeaderBase = coreHeaderRegionStart + blockIdx * CORE_HEADER_SIZE;
    gt.SetValue(coreHeaderBase, static_cast<int64_t>(iter + 1));
}

// ============================================================
// Conditional profiling macros
// ============================================================

#ifdef ASCEND_PROFILE_ENABLE
  #define PROF_INIT(gt)          ProfileInit(gt)
  #define PROF_RECORD_TIME(tag)  RecordTime(tag)

  // PROF_RECORD_TIME_SYNC(tag)         → PipeBarrier<PIPE_ALL>() + RecordTime
  // PROF_RECORD_TIME_SYNC(tag, PIPE_V) → PipeBarrier<PIPE_V>()  + RecordTime
  #define _PROF_RTS_1(tag)                RecordTimeSync<>(tag)
  #define _PROF_RTS_2(tag, pipe)          RecordTimeSync<pipe>(tag)
  #define _PROF_RTS_SEL(_1, _2, NAME, ...) NAME
  #define PROF_RECORD_TIME_SYNC(...)      _PROF_RTS_SEL(__VA_ARGS__, _PROF_RTS_2, _PROF_RTS_1)(__VA_ARGS__)

  #define PROF_TO_GM(gt)         ProfileToGm(gt)
  #define PROF_SLEEP_US(us)      SleepUs(us)
  #define PROF_SYNC_ALL()        AscendC::SyncAll<true>()
#else
  #define PROF_INIT(gt)          ((void)0)
  #define PROF_RECORD_TIME(tag)  ((void)0)
  #define PROF_RECORD_TIME_SYNC(...)  ((void)0)
  #define PROF_TO_GM(gt)         ((void)0)
  #define PROF_SLEEP_US(us)      ((void)0)
  #define PROF_SYNC_ALL()        ((void)0)
#endif

// ============================================================
// Helper: calculate required GM buffer size (bytes) for host allocation
// ============================================================
// GM base assumed 64B aligned.
// Layout: GlobalHeader + coreNum × CoreHeader + maxIters × coreNum × IterCoreStride
#define PROF_GM_BUF_SIZE(coreNum, maxIters) \
    (static_cast<int64_t>(AscendProf::GLOBAL_HEADER_SIZE \
        + (coreNum) * AscendProf::CORE_HEADER_SIZE \
        + (maxIters) * (coreNum) * AscendProf::ITER_CORE_STRIDE) \
     * static_cast<int64_t>(sizeof(int64_t)))

#endif // KERNEL_TOOL_H
