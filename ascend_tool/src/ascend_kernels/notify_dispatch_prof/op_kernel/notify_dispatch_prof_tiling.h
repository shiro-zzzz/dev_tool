/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: notify dispatch prof tiling header file
 * Create: 2026-01-05
 * Note:
 * History: 2026-01-05 create notify dispatch prof tiling header file
 */

#ifndef NOTIFY_DISPATCH_PROF_TILING_H
#define NOTIFY_DISPATCH_PROF_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace Cam {
struct NotifyDispatchProfInfo {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t localRankSize;
    uint32_t localRankId;
    uint32_t sendCount;
    uint32_t numTokens;
    uint32_t aivNum;
    uint64_t totalUbSize;
};

struct NotifyDispatchProfTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    NotifyDispatchProfInfo notifyDispatchInfo;
};
} // namespace Cam
#endif // NOTIFY_DISPATCH_PROF_TILING_H
