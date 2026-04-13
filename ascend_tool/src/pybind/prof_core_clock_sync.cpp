/*
 * prof_core_clock_sync.cpp — PyTorch pybind wrapper for ProfCoreClockSync
 *
 * Calls the CANN custom operator via EXEC_NPU_CMD macro.
 */

#include "pytorch_extension/pytorch_npu_helper.hpp"

at::Tensor ProfCoreClockSyncImpl(const at::Tensor &sync_buf)
{
    auto sync_timestamps = at::empty_like(sync_buf);
    EXEC_NPU_CMD(aclnnProfCoreClockSync, sync_buf, sync_timestamps);
    return sync_timestamps;
}
