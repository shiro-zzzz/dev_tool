# 测试

本目录包含项目的测试代码。

## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_example.py

# 查看覆盖率
pytest --cov=src tests/
```

## 测试结构

```
tests/
├── test_unit/          # 单元测试
├── test_integration/   # 集成测试
└── test_e2e/          # 端到端测试
```

## 编写测试

请确保：
- 每个功能都有对应的测试
- 测试命名清晰
- 包含边界情况测试
- 测试相互独立
