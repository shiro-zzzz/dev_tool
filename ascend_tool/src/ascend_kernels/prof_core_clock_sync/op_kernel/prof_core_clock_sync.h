/*
 * prof_core_clock_sync.h — Ascend C kernel: single-card multi-core clock sync
 *
 * Operator A: ProfCoreClockSync
 *
 * Synchronizes all AICore clocks within one NPU card by performing multiple
 * rounds of SyncAll barriers followed by immediate timestamp capture.
 *
 * GM Output Layout (int64_t units):
 *
 * ┌────────────────────────────────────────────────────────────────────────┐
 * │  Header (8 × int64_t, written by core 0 only):                       │
 * │  [0] blockNum        — number of participating cores                  │
 * │  [1] syncRounds      — number of sync rounds performed                │
 * │  [2] coreStride      — int64 stride between core regions             │
 * │  [3] coreRegionStart — first core region offset (cacheline aligned)   │
 * │  [4..7] reserved                                                      │
 * ├────────────────────────────────────────────────────────────────────────┤
 * │  Per-Core Region (at coreRegionStart + i × coreStride):              │
 * │  +0 .. +(syncRounds-1) : timestamps (cycle count) per sync round     │
 * │  Padded to cacheline boundary (8 int64 aligned)                       │
 * └────────────────────────────────────────────────────────────────────────┘
 *
 * Host offset calculation:
 *   offset_i = mean(ts_i[r] - ts_0[r]) for r in [0, syncRounds)
 *
 * Required GM buffer size (int64_t count):
 *   PROF_CORE_SYNC_BUF_INT64(coreNum, syncRounds)
 */

#ifndef PROF_CORE_CLOCK_SYNC_H
#define PROF_CORE_CLOCK_SYNC_H

#include "kernel_operator.h"

namespace ProfCoreClockSyncImpl {

using namespace AscendC;

constexpr int32_t CACHELINE_BYTES = 64;
constexpr int32_t CACHELINE_INT64 = CACHELINE_BYTES / static_cast<int32_t>(sizeof(int64_t)); // 8
constexpr int32_t HEADER_SIZE = CACHELINE_INT64; // 8

__aicore__ constexpr int32_t AlignUpCL(int32_t x)
{
    return ((x + CACHELINE_INT64 - 1) / CACHELINE_INT64) * CACHELINE_INT64;
}

class ProfCoreClockSync {
public:
    __aicore__ inline ProfCoreClockSync() {}

    __aicore__ inline void Init(GM_ADDR syncBufGM, GM_ADDR outputGM,
                                uint32_t syncRounds)
    {
        syncRounds_ = syncRounds;
        blockIdx_ = static_cast<int32_t>(GetBlockIdx());
        blockNum_ = static_cast<int32_t>(GetBlockNum());
        coreStride_ = AlignUpCL(static_cast<int32_t>(syncRounds_));

        outputGT_.SetGlobalBuffer((__gm__ int64_t *)outputGM);

        // GM base assumed 64B aligned, coreRegionStart = HEADER_SIZE
        coreRegionStart_ = HEADER_SIZE;
    }

    __aicore__ inline void Process()
    {
        // Write header (core 0 only)
        if (blockIdx_ == 0) {
            outputGT_.SetValue(0, static_cast<int64_t>(blockNum_));
            outputGT_.SetValue(1, static_cast<int64_t>(syncRounds_));
            outputGT_.SetValue(2, static_cast<int64_t>(coreStride_));
            outputGT_.SetValue(3, static_cast<int64_t>(coreRegionStart_));
            outputGT_.SetValue(4, 0LL);
            outputGT_.SetValue(5, 0LL);
            outputGT_.SetValue(6, 0LL);
            outputGT_.SetValue(7, 0LL);
        }

        int32_t coreBase = coreRegionStart_ + blockIdx_ * coreStride_;

        // Multiple rounds of synchronized timestamp capture
        for (uint32_t round = 0; round < syncRounds_; round++) {
            // Barrier 1: all cores reach this point
            SyncAll<true>();
            // Barrier 2: eliminates GM-write latency carryover from previous
            // round — all cores are guaranteed idle before timestamp capture
            SyncAll<true>();

            // Immediately capture timestamp after double barrier
            int64_t cycle = static_cast<int64_t>(GetSystemCycle());

            // Write to per-core slot for this round
            outputGT_.SetValue(coreBase + static_cast<int32_t>(round), cycle);

            // Ensure write completes before next barrier
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    GlobalTensor<int64_t> outputGT_;
    uint32_t syncRounds_{0};
    int32_t blockIdx_{0};
    int32_t blockNum_{0};
    int32_t coreStride_{0};
    int32_t coreRegionStart_{0};
};

} // namespace ProfCoreClockSyncImpl

/// Required GM buffer size in int64_t units (GM base assumed 64B aligned)
#define PROF_CORE_SYNC_BUF_INT64(coreNum, syncRounds) \
    (ProfCoreClockSyncImpl::HEADER_SIZE \
     + (coreNum) * ProfCoreClockSyncImpl::AlignUpCL(static_cast<int32_t>(syncRounds)))

/// Required GM buffer size in bytes
#define PROF_CORE_SYNC_BUF_SIZE(coreNum, syncRounds) \
    (static_cast<int64_t>(PROF_CORE_SYNC_BUF_INT64(coreNum, syncRounds)) \
     * static_cast<int64_t>(sizeof(int64_t)))

#endif // PROF_CORE_CLOCK_SYNC_H
