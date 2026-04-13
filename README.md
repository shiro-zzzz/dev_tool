# dev_toolkit

> 一个开发工具包项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 简介

dev_toolkit 是一个开发工具集合，旨在提高开发效率。

## 🧰 工具导航

### 1) Ascend 打点工具（优先推荐）

- 文档入口：[`ascend_tool/README_kernel_profiler.md`](ascend_tool/README_kernel_profiler.md)
- 功能简介：用于 Ascend 自定义算子执行链路打点与耗时分析，支持定位算子内核阶段的性能瓶颈，便于快速做性能优化与问题排查。

### 2) lcov 映射工具

- 文档入口：[`README_lcov_mapper.md`](README_lcov_mapper.md)
- 功能简介：用于处理与映射 lcov 覆盖率结果，帮助将覆盖率数据和源码路径对齐，便于在多目录或生成代码场景下查看覆盖率。

## ✨ 特性

- 🚀 易于使用
- 📦 模块化设计
- 🔧 可扩展
- 📝 文档完善

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/dev_toolkit.git
cd dev_toolkit

# 安装依赖
# TODO: 添加安装命令
```

### 使用示例

```bash
# TODO: 添加使用示例
```

## 📁 项目结构

```
dev_toolkit/
├── ascend_tool/                      # Ascend 打点工具
│   ├── README.md
│   ├── README_kernel_profiler.md     # 打点工具详细文档
│   ├── examples/                     # 打点示例与测试
│   └── src/                          # 打点工具实现与算子代码
│       ├── ascend_prof_tool.py       # 打点工具主入口
│       ├── ascend_kernels/           # Ascend 自定义算子与生成代码
│       │   ├── notify_dispatch_prof/
│       │   ├── vec_add_prof/
│       │   ├── prof_core_clock_sync/
│       │   ├── utils/                # host/kernel 通用工具
│       │   ├── pregen/               # aclnn 自动生成产物
│       │   └── cmake_files/
│       └── pybind/                   # Python 绑定与扩展打包
│           └── pytorch_extension/
├── docs/                             # 通用文档
├── tests/                            # 通用测试
├── examples/                         # 通用示例
├── README_lcov_mapper.md             # lcov 映射工具文档
├── README.md                         # 项目说明
└── .cursor/                          # Cursor 配置与技能
```

## 🤝 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📮 联系方式

- 提交 Issue: [GitHub Issues](https://github.com/yourusername/dev_toolkit/issues)
- 邮件: your.email@example.com

## 🙏 致谢

感谢所有贡献者！