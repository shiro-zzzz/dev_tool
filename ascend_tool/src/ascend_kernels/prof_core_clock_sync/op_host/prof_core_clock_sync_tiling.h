/*
 * prof_core_clock_sync_tiling.h — shared tiling data between host and kernel
 *
 * Operator A: ProfCoreClockSync — Single-card multi-core clock synchronization
 */

#ifndef PROF_CORE_CLOCK_SYNC_TILING_H
#define PROF_CORE_CLOCK_SYNC_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ProfCoreClockSyncTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, syncRounds);   // Number of sync rounds for averaging
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);      // Number of AI cores participating
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ProfCoreClockSync, ProfCoreClockSyncTilingData)
} // namespace optiling

#endif // PROF_CORE_CLOCK_SYNC_TILING_H
