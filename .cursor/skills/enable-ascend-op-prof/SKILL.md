---
name: enable-ascend-op-prof
description: Add profiling support to an Ascend custom operator end-to-end by wiring prof_buf through op_host/json/kernel/pybind/aclnn and generating trace files in torch tests. Use when user asks to add prof_buf, kernel profiling tags, pybind signature updates, or trace JSON output for an operator.
---

# Enable Ascend Op Profiling

## 目标

给任意 Ascend 自定义算子增加 `prof_buf` 打点能力，并在 torch 用例中输出可视化 trace 文件（单 rank + merged）。

## 适用触发词

出现以下需求时使用本技能：

- “给某算子加 `prof_buf` 入参”
- “参考 VecAddProf 给 kernel 打点”
- “pybind / aclnn 签名同步”
- “生成 trace json / Perfetto 文件”
- “`parse_prof_buf` 越界 / index out of bounds”

## 标准改造流程

### 1) op_host 与 op json：新增 `prof_buf` 输入

需要同步两处：

- `src/ascend_kernels/<OpName>.json`
- `src/ascend_kernels/<op>/op_host/<op>.cpp`

约束：

- `prof_buf` 类型固定 `int64`
- `DataType/Format/UnknownShapeFormat` 的元素数量要和该算子其他输入的“组合组数”一致  
  （同索引表示一组 dtype-layout 组合）

示例（3 组）：

- `DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})`
- `Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})`
- `UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})`

---

### 2) kernel：接入 `prof_buf` 并打点

修改：

- `op_kernel/<op>.cpp`：kernel 入口参数新增 `GM_ADDR profBuf`
- `op_kernel/<op>.h`：类 `Init(...)` 新增 `profBuf`

实现要点：

1. 在 `.h` 里启用打点宏并引入：
   - `#define ASCEND_PROFILE_ENABLE`
   - `#include "kernel_tool.h"`
2. 新增成员：
   - `GlobalTensor<int64_t> profGm;`
3. 在 `Init(...)` 内绑定：
   - `profGm.SetGlobalBuffer((__gm__ int64_t *)profBuf);`
4. 在 `Process()` 里插桩：
   - 开始：`PROF_INIT(profGm);`
   - 阶段完成点：`PROF_RECORD_TIME(tag);`
   - 关键同步点：`PROF_SYNC_ALL();`
   - 结束：`PROF_RECORD_TIME(999); PROF_TO_GM(profGm);`

建议 tag 约定：

- `0`: Init（`PROF_INIT` 自动）
- `100/200/300...`: 阶段完成点
- `999`: AllDone
- `99999`: IterEnd（`PROF_TO_GM` 自动）

---

### 3) pybind：签名与调用链透传 `prof_buf`

必须同步三处：

- `src/pybind/functions.h`
- `src/pybind/<op>.cpp`
- `src/pybind/pybind.cpp`

检查点：

1. C++ 声明增加 `const at::Tensor &profBuf`
2. `EXEC_NPU_CMD(aclnnXxx, ...)` 里把 `profBuf` 放到与算子定义一致的位置
3. `PYBIND11_MODULE` 的 `py::arg("prof_buf")` 加入
4. `TORCH_LIBRARY` schema 同步更新 `Tensor prof_buf`

---

### 4) aclnn 包装层：GetWorkspaceSize 签名同步

文件（autogen 目录）：

- `aclnn_<op>.h`
- `aclnn_<op>.cpp`

要求：

1. `aclnnXxxGetWorkspaceSize(...)` 增加 `const aclTensor *profBuf`
2. 调用 `aclnnInnerXxxGetWorkspaceSize(...)` 时透传 `profBuf`
3. `aclnnXxx(...)` 主执行函数通常不需要改参数（保持 workspace/executor/stream）

---

### 5) torch 用例：生成 trace 文件

推荐流程（多卡）：

1. `tool = AscendProfTool(sync_rounds=16)`
2. 先做 HCCL warmup，并在与业务一致的通信组上调用：
   - `rank_offsets = tool.calibrate_rank_clocks(world_size, rank, group=group)`
3. 根据 `rank_offsets` 推断当前 rank 核数并分配 `prof_buf`
4. 调算子时传入 `prof_buf`
5. `prof_data = AscendProfTool.parse_prof_buf(prof_buf)`
6. 生成单 rank trace：
   - `tool.generate_trace_json(..., rank_id=rank, tag_names=...)`
7. 生成 merged trace：
   - `tool.generate_merged_trace_json(..., world_size=..., rank=..., tag_names=...)`

`tag_names` 示例：

```python
TAG_NAMES = {
    0: "Init",
    100: "StageA",
    200: "StageB",
    999: "AllDone",
    99999: "IterEnd",
}
```

---

## 必做防护（避免 parse 越界）

### A. `prof_buf` 大小不要硬编码旧核数

`kernel_tool.h` 布局（int64 元素）：

`size = 8 + block_dim * 8 + max_iters * block_dim * 136`

如果实际核数是 48，却按 32 分配，会在 `parse_prof_buf` 读取 tag 时越界。

### B. parser 加边界保护

在 `parse_prof_buf` 中：

- 若 `base >= len(buf)`：跳过
- `cnt` 截断到 `min(max_slots, (iter_core_stride - core_meta_size)//2)`
- 每次读取 `tag/ts` 前检查索引是否越界

---

## 自检清单

- [ ] op_host/json 都有 `prof_buf`
- [ ] `DataType/Format/UnknownShapeFormat` 组数一致
- [ ] kernel 入口 + `Init` + `GlobalTensor<int64_t>` 全链路透传
- [ ] `PROF_INIT` 与 `PROF_TO_GM` 成对出现
- [ ] pybind 三处同步（声明/实现/schema）
- [ ] aclnn GetWorkspaceSize 签名与透传一致
- [ ] 用例中 `prof_buf` 按实际核数分配
- [ ] 生成了 rank trace 与 merged trace

## 常见故障定位

1. `IndexError: ... out of bounds`
   - 首查 `prof_buf` 分配核数是否与实际 `block_num` 一致
2. 运行期接口不匹配
   - 检查 pybind schema 与 `EXEC_NPU_CMD` 参数顺序
3. 算子注册或编译异常
   - 检查 json 与 op_host 的参数列表是否同步
4. trace 名称不可读
   - 补 `tag_names` 映射
