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
    m.def("notify_dispatch_prof", &NotifyDispatchProfImpl, "notify_dispatch_prof",
          py::arg("send_data"), py::arg("token_per_expert_data"),
          py::arg("send_count"), py::arg("num_tokens"),
          py::arg("comm_group"), py::arg("rank_size"), py::arg("rank_id"),
          py::arg("local_rank_size"), py::arg("local_rank_id"));
}

TORCH_LIBRARY(ascend_tool, m)
{
    m.def("vec_add_prof(Tensor x, Tensor y, Tensor prof_buf) -> Tensor");
    m.def("prof_core_clock_sync(Tensor sync_buf) -> Tensor");
    m.def("notify_dispatch_prof(Tensor send_data, Tensor token_per_expert_data, int send_count, int num_tokens, str comm_group, int rank_size, int rank_id, int local_rank_size, int local_rank_id) -> Tensor[]");
}