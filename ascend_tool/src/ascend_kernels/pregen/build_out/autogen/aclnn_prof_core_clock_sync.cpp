/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: add prof core clock sync interface cpp file.
 * Create: 2026-04-01
 * Note:
 * History: 2026-04-01 add prof core clock sync interface cpp file.
 */

#include <string.h>
#include "aclnnInner_prof_core_clock_sync.h"
#include "graph/types.h"
#include "aclnn_prof_core_clock_sync.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnProfCoreClockSyncGetWorkspaceSize(const aclTensor *syncBuf, const aclTensor *syncTimestamps,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerProfCoreClockSyncGetWorkspaceSize(syncBuf, syncTimestamps, workspaceSize, executor);
}

aclnnStatus aclnnProfCoreClockSync(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    return aclnnInnerProfCoreClockSync(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
