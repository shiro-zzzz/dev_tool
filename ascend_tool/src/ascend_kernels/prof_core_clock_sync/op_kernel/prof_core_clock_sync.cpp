/*
 * prof_core_clock_sync.cpp — Ascend C kernel entry: single-card multi-core clock sync
 *
 * Operator A: ProfCoreClockSync
 */

#include "kernel_operator.h"
#include "prof_core_clock_sync.h"

using namespace AscendC;
using namespace ProfCoreClockSyncImpl;

extern "C" __global__ __aicore__ void
prof_core_clock_sync(GM_ADDR syncBuf, GM_ADDR syncTimestamps, GM_ADDR workspace, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    ProfCoreClockSync op;
    op.Init(syncBuf, syncTimestamps, tilingData.syncRounds);
    op.Process();
}
