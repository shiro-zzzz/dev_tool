# ascend_tool (CAM comm_operator build framework)

这个目录包含两部分：

1) “编译框架代码”（已迁移到 `ascend_tool/src`）
2) “算子实现代码”（默认仍保留在 `umdk_gitcode/src/cam/comm_operator`）

编译时会把算子实现的必要 `.cpp` / `op_host` / `op_kernel` 临时拷贝进构建目录，从而只需要“添加实现文件”就能生成 run 包和 pybind whl。

## 可配置项

- `UMDK_GITCODE_ROOT`：`umdk_gitcode` 仓库根目录（默认：`../..` 推导到 `umdk_gitcode`）
- `CAM_SRC_BASE_PATH`：框架目录基路径（默认：`ascend_tool/src`），其下应包含 `ascend_kernels/` 和 `pybind/`
- `CAM_OP_IMPL_BASE_PATH`：算子实现目录基路径（默认：`${UMDK_GITCODE_ROOT}/src/cam/comm_operator`）

## 常用命令

运行 run 包 + pybind whl（默认都会做）：

```bash
bash ./ascend_tool/build.sh comm_operator
```

只打 pybind whl（跳过 run 包编译）：

```bash
bash ./ascend_tool/build.sh comm_operator -p
```

抽取 run 包内容：

```bash
bash ./ascend_tool/build.sh comm_operator -x
```

## 输出目录

- run 包：`ascend_tool/output/comm_operator/run/`
- 抽取内容：`ascend_tool/output/comm_operator/extract/<soc>/`
- whl：`ascend_tool/output/comm_operator/dist/`

