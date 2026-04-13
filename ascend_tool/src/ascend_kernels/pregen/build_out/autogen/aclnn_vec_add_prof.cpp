/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: add vec add prof interface cpp file.
 * Create: 2026-04-01
 * Note:
 * History: 2026-04-01 add vec add prof interface cpp file.
 */

#include <string.h>
#include "aclnnInner_vec_add_prof.h"
#include "graph/types.h"
#include "aclnn_vec_add_prof.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnVecAddProfGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *profBuf,
                                             const aclTensor *z, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerVecAddProfGetWorkspaceSize(x, y, profBuf, z, workspaceSize, executor);
}

aclnnStatus aclnnVecAddProf(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerVecAddProf(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
