/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: add pybind
 * Create: 2025-12-10
 * Note:
 * History: 2025-12-10 add pybind
 */

#include "functions.h"
#include <torch/extension.h>
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vec_add_prof", &VecAddProfImplAutograd, "vec_add_prof");
    m.def("prof_core_clock_sync", &ProfCoreClockSyncImpl, "prof_core_clock_sync");
}

TORCH_LIBRARY(ascend_tool, m)
{
    m.def("vec_add_prof(Tensor x, Tensor y, Tensor prof_buf) -> Tensor");
    m.def("prof_core_clock_sync(Tensor sync_buf) -> Tensor");
}