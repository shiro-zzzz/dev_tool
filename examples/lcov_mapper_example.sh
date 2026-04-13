#!/bin/bash
# LCOV 路径映射工具使用示例

echo "=== 示例 1: 基本用法 ==="
# 将 coverage.info 中的路径映射到 ./src 目录
python3 ../src/lcov_path_mapper.py \
    -i coverage.info \
    -o coverage_fixed.info \
    -s ./src

echo ""
echo "=== 示例 2: 使用详细输出模式 ==="
# 查看详细的处理过程
python3 ../src/lcov_path_mapper.py \
    -i coverage.info \
    -o coverage_fixed.info \
    -s /path/to/real/source \
    -v

echo ""
echo "=== 示例 3: 完整工作流（映射 + 生成报告） ==="
# 先映射路径
python3 ../src/lcov_path_mapper.py \
    -i coverage.info \
    -o coverage_fixed.info \
    -s ./src \
    -v

# 然后使用修正后的 info 文件生成 HTML 报告
genhtml coverage_fixed.info \
    --output-directory coverage_report \
    --title "项目代码覆盖率报告" \
    --show-details \
    --legend

echo ""
echo "处理完成！可以打开 coverage_report/index.html 查看报告"

echo ""
echo "=== 示例 4: 使用绝对路径 ==="
# 使用绝对路径处理
python3 ../src/lcov_path_mapper.py \
    -i /path/to/build/coverage.info \
    -o /path/to/output/coverage_fixed.info \
    -s /path/to/project/src

echo ""
echo "=== 示例 5: 合并多个有效的 info 文件 ==="

# 定义函数：遍历、验证并合并 info 文件
merge_valid_lcov_files() {
    local search_dir="${1:-.}"      # 搜索目录，默认为当前目录
    local output_file="${2:-merged_coverage.info}"  # 输出文件名
    local temp_dir=$(mktemp -d)
    local valid_files=()
    local invalid_files=()
    
    echo "开始在目录 '$search_dir' 中搜索 .info 文件..."
    echo ""
    
    # 遍历查找所有 .info 文件
    while IFS= read -r -d '' info_file; do
        echo "检查文件: $info_file"
        
        # 检查文件是否可读且非空
        if [ ! -r "$info_file" ]; then
            echo "  ✗ 文件不可读或不存在"
            invalid_files+=("$info_file")
            echo ""
            continue
        fi
        
        if [ ! -s "$info_file" ]; then
            echo "  ✗ 文件为空"
            invalid_files+=("$info_file")
            echo ""
            continue
        fi
        
        # 检查文件基本结构
        local has_tn=$(grep -c "^TN:" "$info_file" 2>/dev/null || echo 0)
        local has_sf=$(grep -c "^SF:" "$info_file" 2>/dev/null || echo 0)
        local has_eor=$(grep -c "^end_of_record" "$info_file" 2>/dev/null || echo 0)
        local has_da=$(grep -c "^DA:" "$info_file" 2>/dev/null || echo 0)
        local has_brda=$(grep -c "^BRDA:" "$info_file" 2>/dev/null || echo 0)
        
        echo "  文件结构: TN:$has_tn SF:$has_sf DA:$has_da BRDA:$has_brda EOR:$has_eor"
        
        # 使用 lcov 验证文件是否有效
        local lcov_error=$(lcov --summary "$info_file" 2>&1)
        local lcov_ret=$?
        
        if [ $lcov_ret -eq 0 ]; then
            # 进一步检查文件是否有实际的覆盖率数据
            if [ $has_da -gt 0 ] || [ $has_brda -gt 0 ]; then
                echo "  ✓ 有效文件，包含覆盖率数据"
                valid_files+=("$info_file")
            else
                echo "  ✗ 文件格式正确但无覆盖率数据"
                invalid_files+=("$info_file")
            fi
        else
            echo "  ✗ lcov 验证失败"
            # 输出错误详情（仅显示前3行）
            echo "  错误信息: $(echo "$lcov_error" | head -n 3 | tr '\n' ' ')"
            
            # 尝试诊断常见问题
            if [ $has_sf -eq 0 ]; then
                echo "  可能原因: 缺少 SF: (源文件路径) 行"
            elif [ $has_eor -eq 0 ]; then
                echo "  可能原因: 缺少 end_of_record 标记"
            elif [ $has_tn -eq 0 ]; then
                echo "  可能原因: 缺少 TN: (测试名称) 行"
            fi
            
            invalid_files+=("$info_file")
        fi
        echo ""
    done < <(find "$search_dir" -type f -name "*.info" -print0)
    
    # 输出统计信息
    echo "========================================"
    echo "扫描完成！"
    echo "有效文件数: ${#valid_files[@]}"
    echo "无效文件数: ${#invalid_files[@]}"
    echo "========================================"
    echo ""
    
    # 如果有无效文件，打印详细列表
    if [ ${#invalid_files[@]} -gt 0 ]; then
        echo "无效文件列表:"
        for file in "${invalid_files[@]}"; do
            echo "  - $file"
        done
        echo ""
    fi
    
    # 合并有效文件
    if [ ${#valid_files[@]} -eq 0 ]; then
        echo "错误: 没有找到有效的 .info 文件"
        rm -rf "$temp_dir"
        return 1
    elif [ ${#valid_files[@]} -eq 1 ]; then
        echo "只找到一个有效文件，直接复制..."
        cp "${valid_files[0]}" "$output_file"
        echo "完成: $output_file"
    else
        echo "开始合并 ${#valid_files[@]} 个有效文件..."
        
        # 构建 lcov 合并命令
        local lcov_cmd="lcov"
        for file in "${valid_files[@]}"; do
            lcov_cmd="$lcov_cmd --add-tracefile $file"
        done
        lcov_cmd="$lcov_cmd --output-file $output_file"
        
        # 执行合并
        if eval "$lcov_cmd" 2>&1; then
            echo ""
            echo "✓ 成功合并到: $output_file"
            echo ""
            echo "合并后的覆盖率摘要:"
            lcov --summary "$output_file"
        else
            echo "✗ 合并失败"
            rm -rf "$temp_dir"
            return 1
        fi
    fi
    
    rm -rf "$temp_dir"
    return 0
}

# 使用示例：
# 在当前目录及子目录下查找并合并
# merge_valid_lcov_files . merged_coverage.info

# 在指定目录下查找并合并
# merge_valid_lcov_files /path/to/coverage/dir output.info

# 使用默认参数（当前目录，输出为 merged_coverage.info）
# merge_valid_lcov_files

echo "函数 merge_valid_lcov_files 已定义"
echo "用法: merge_valid_lcov_files [搜索目录] [输出文件名]"
echo "示例: merge_valid_lcov_files ./coverage merged_result.info"

echo ""
echo "=== 示例 6: 转换 info 文件中的相对路径为绝对路径 ==="

# 简单函数：将 info 文件中包含 "../" 的路径转换为绝对路径
normalize_lcov_paths() {
    local input_file="$1"
    local output_file="${2:-${input_file%.info}_normalized.info}"
    
    if [ ! -f "$input_file" ]; then
        echo "错误: 文件不存在: $input_file"
        return 1
    fi
    
    local temp_file=$(mktemp)
    local converted=0
    
    while IFS= read -r line; do
        # 检查是否是 SF: 开头且包含 "../"
        if [[ "$line" =~ ^SF:(.+)$ ]] && [[ "$line" == *"../"* ]]; then
            local path="${BASH_REMATCH[1]}"
            # 使用 Python 转换为绝对路径
            local abs_path=$(python3 -c "import os; print(os.path.abspath('$path'))" 2>/dev/null)
            if [ -n "$abs_path" ]; then
                echo "SF:$abs_path" >> "$temp_file"
                echo "转换: $path -> $abs_path"
                converted=$((converted + 1))
            else
                echo "$line" >> "$temp_file"
            fi
        else
            echo "$line" >> "$temp_file"
        fi
    done < "$input_file"
    
    mv "$temp_file" "$output_file"
    echo "完成: 转换 $converted 个路径，输出到 $output_file"
}

# 使用示例：
# normalize_lcov_paths coverage.info
# normalize_lcov_paths coverage.info output.info

echo "函数 normalize_lcov_paths 已定义"
echo "用法: normalize_lcov_paths <input.info> [output.info]"