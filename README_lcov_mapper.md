# LCOV 路径映射工具

## 问题背景

在使用 lcov 进行代码覆盖率分析时，如果源码是通过拷贝到其他目录进行编译的，生成的 `coverage.info` 文件中的路径会指向拷贝后的位置，而不是真正的源码位置。这会导致：

- 无法正确关联到源码仓库
- 生成的 HTML 报告指向错误的路径
- 持续集成中无法正确展示覆盖率

## 解决方案

本工具通过以下方式解决路径映射问题：

1. **文件名匹配**: 在指定的源码目录中查找同名文件
2. **哈希验证**: 对于同名文件，通过 MD5 哈希值确保内容一致
3. **路径替换**: 将 info 文件中的路径替换为真实源码的绝对路径

## 使用方法

### 基本用法

```bash
python3 src/lcov_path_mapper.py \
    -i coverage.info \
    -o coverage_fixed.info \
    -s /path/to/real/source
```

### 参数说明

- `-i, --input`: 输入的 lcov info 文件路径（必需）
- `-o, --output`: 输出的修正后的 info 文件路径（必需）
- `-s, --source-dir`: 真正的源码目录路径（必需）
- `-v, --verbose`: 输出详细的处理信息（可选）

### 示例

#### 示例 1: 映射到当前项目的 src 目录

```bash
python3 src/lcov_path_mapper.py \
    -i build/coverage.info \
    -o coverage_fixed.info \
    -s ./src
```

#### 示例 2: 查看详细处理过程

```bash
python3 src/lcov_path_mapper.py \
    -i coverage.info \
    -o coverage_fixed.info \
    -s /home/user/project/src \
    -v
```

输出示例：
```
正在扫描源码目录: /home/user/project/src
找到 156 个文件，142 个不同的文件名
正在处理: coverage.info

处理文件: /tmp/build/src/main.c
  找到 2 个同名文件，使用哈希匹配
  哈希匹配成功: /home/user/project/src/main.c
✓ main.c -> /home/user/project/src/main.c

处理文件: /tmp/build/src/utils.c
  找到唯一匹配: /home/user/project/src/utils.c
✓ utils.c -> /home/user/project/src/utils.c

处理完成:
  总文件数: 45
  成功映射: 43
  未能映射: 2
  输出文件: coverage_fixed.info
```

#### 示例 3: 完整工作流

```bash
# 1. 运行测试并生成覆盖率数据
make test-coverage

# 2. 使用 lcov 收集覆盖率信息
lcov --capture --directory build --output-file coverage.info

# 3. 修正路径
python3 src/lcov_path_mapper.py \
    -i coverage.info \
    -o coverage_fixed.info \
    -s ./src

# 4. 生成 HTML 报告
genhtml coverage_fixed.info --output-directory coverage_report

# 5. 查看报告
open coverage_report/index.html
```

## 工作原理

### 1. 文件索引构建

脚本首先扫描源码目录，建立文件名到路径的映射：

```
文件名索引:
  main.c -> [/src/main.c, /src/legacy/main.c]
  utils.c -> [/src/utils.c]
  config.c -> [/src/config.c]
```

### 2. 路径匹配

对于 info 文件中的每个 `SF:` 行（源文件路径）：

- 提取文件名
- 在索引中查找同名文件
- 如果只有一个匹配，直接使用
- 如果有多个匹配，计算哈希值进行精确匹配

### 3. 哈希验证

当存在多个同名文件时：

```python
# 计算编译文件的哈希
compiled_hash = md5(/tmp/build/src/main.c)

# 计算候选文件的哈希
for candidate in candidates:
    if md5(candidate) == compiled_hash:
        return candidate  # 找到匹配
```

### 4. 路径替换

找到匹配后，替换 info 文件中的路径：

```
# 原始
SF:/tmp/build/src/main.c

# 修正后
SF:/home/user/project/src/main.c
```

## 注意事项

1. **文件内容必须一致**: 工具通过哈希匹配，确保源码未被修改
2. **性能考虑**: 首次运行会扫描整个源码目录，大型项目可能需要几秒钟
3. **路径格式**: 输出的路径是绝对路径
4. **编码支持**: 支持 UTF-8 编码的文件

## 故障排查

### 问题: 部分文件未能映射

```
✗ 未找到匹配: /tmp/build/generated/config.h
```

**可能原因**:
- 文件是生成的，不在源码目录中
- 文件内容已被修改
- 文件名不同

**解决方案**:
- 使用 `-v` 参数查看详细信息
- 确认源码目录是否包含该文件
- 检查文件内容是否一致

### 问题: 哈希匹配失败

```
警告: 未找到哈希匹配的文件
```

**可能原因**:
- 编译过程修改了文件内容（如预处理）
- 文件编码不同
- 行尾符差异 (CRLF vs LF)

**解决方案**:
- 检查编译配置
- 确保源文件未被修改

## 高级用法

### 集成到 Makefile

```makefile
coverage-fix:
	lcov --capture --directory build --output-file coverage.info
	python3 src/lcov_path_mapper.py -i coverage.info -o coverage_fixed.info -s ./src
	genhtml coverage_fixed.info -o coverage_report

.PHONY: coverage-fix
```

### 集成到 CI/CD

```yaml
# GitHub Actions 示例
- name: Fix coverage paths
  run: |
    python3 src/lcov_path_mapper.py \
      -i coverage.info \
      -o coverage_fixed.info \
      -s ${{ github.workspace }}/src
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage_fixed.info
```

## 许可证

MIT License
