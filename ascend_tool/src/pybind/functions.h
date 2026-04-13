/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: pybind functions header file
 * Create: 2025-12-10
 * Note:
 * History: 2025-12-10 add pybind functions header file
 */

#ifndef COMMON_OPS_CSRC_FUNCTIONS_H_
#define COMMON_OPS_CSRC_FUNCTIONS_H_

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include <torch/script.h>

at::Tensor VecAddProfImplAutograd(
    const at::Tensor &x,
    const at::Tensor &y,
    const at::Tensor &prof_buf);

at::Tensor ProfCoreClockSyncImpl(const at::Tensor &sync_buf);

#endif // COMMON_OPS_CSRC_FUNCTIONS_H_