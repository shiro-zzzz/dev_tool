/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: add vec add prof interface header file.
 * Create: 2026-04-01
 * Note:
 * History: 2026-04-01 add vec add prof interface header file.
 */

#ifndef ACLNN_VEC_ADD_PROF_H_
#define ACLNN_VEC_ADD_PROF_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnVecAddProfGetWorkspaceSize(
    const aclTensor *x, const aclTensor *y, const aclTensor *profBuf,
    const aclTensor *z, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnVecAddProf(void *workspace, uint64_t workspaceSize,
                                                                    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
