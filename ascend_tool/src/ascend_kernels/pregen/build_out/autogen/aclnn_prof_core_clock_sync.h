/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: add prof core clock sync interface header file.
 * Create: 2026-04-01
 * Note:
 * History: 2026-04-01 add prof core clock sync interface header file.
 */

#ifndef ACLNN_PROF_CORE_CLOCK_SYNC_H_
#define ACLNN_PROF_CORE_CLOCK_SYNC_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnProfCoreClockSyncGetWorkspaceSize(
    const aclTensor *syncBuf, const aclTensor *syncTimestamps,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnProfCoreClockSync(void *workspace, uint64_t workspaceSize,
                                                                          aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
