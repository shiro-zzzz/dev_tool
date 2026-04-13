#!/usr/bin/env python3
"""
LCOV 路径映射工具

用于修正 lcov 生成的 info 文件中的源文件路径。
当源码被拷贝到其他目录编译时，info 文件中的路径会指向拷贝位置，
此工具可以将路径映射回真正的源码位置。
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional


class LcovPathMapper:
    """LCOV 路径映射器"""
    
    def __init__(self, source_dir: str, verbose: bool = False):
        """
        初始化路径映射器
        
        Args:
            source_dir: 真正的源码目录
            verbose: 是否输出详细信息
        """
        self.source_dir = Path(source_dir).resolve()
        self.verbose = verbose
        # 文件哈希缓存: {文件路径: 哈希值}
        self.hash_cache: Dict[Path, str] = {}
        # 文件名索引: {文件名: [路径列表]}
        self.filename_index: Dict[str, List[Path]] = {}
        
    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """
        计算文件的 MD5 哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的 MD5 哈希值，如果文件不存在或无法读取则返回 None
        """
        if file_path in self.hash_cache:
            return self.hash_cache[file_path]
        
        try:
            if not file_path.exists() or not file_path.is_file():
                return None
                
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            file_hash = hash_md5.hexdigest()
            self.hash_cache[file_path] = file_hash
            return file_hash
        except Exception as e:
            if self.verbose:
                print(f"警告: 无法读取文件 {file_path}: {e}", file=sys.stderr)
            return None
    
    def _build_filename_index(self):
        """构建源码目录中的文件名索引"""
        if self.verbose:
            print(f"正在扫描源码目录: {self.source_dir}")
        
        if not self.source_dir.exists():
            raise ValueError(f"源码目录不存在: {self.source_dir}")
        
        # 遍历源码目录，建立文件名到路径的映射
        for file_path in self.source_dir.rglob('*'):
            if file_path.is_file():
                filename = file_path.name
                if filename not in self.filename_index:
                    self.filename_index[filename] = []
                self.filename_index[filename].append(file_path)
        
        if self.verbose:
            total_files = sum(len(paths) for paths in self.filename_index.values())
            print(f"找到 {total_files} 个文件，{len(self.filename_index)} 个不同的文件名")
    
    def find_matching_source(self, compiled_path: Path) -> Optional[Path]:
        """
        查找与编译路径匹配的真实源文件
        
        Args:
            compiled_path: 编译时的文件路径
            
        Returns:
            匹配的源文件路径，如果未找到则返回 None
        """
        filename = compiled_path.name
        
        # 如果文件名不在索引中，说明没有匹配的文件
        if filename not in self.filename_index:
            if self.verbose:
                print(f"  未找到文件名: {filename}")
            return None
        
        candidate_paths = self.filename_index[filename]
        
        # 如果只有一个候选文件，直接返回
        if len(candidate_paths) == 1:
            if self.verbose:
                print(f"  找到唯一匹配: {candidate_paths[0]}")
            return candidate_paths[0]
        
        # 有多个同名文件，通过哈希值匹配
        if self.verbose:
            print(f"  找到 {len(candidate_paths)} 个同名文件，使用哈希匹配")
        
        compiled_hash = self._calculate_file_hash(compiled_path)
        if compiled_hash is None:
            if self.verbose:
                print(f"  警告: 无法计算编译文件的哈希: {compiled_path}")
            return None
        
        # 查找哈希值匹配的文件
        for candidate_path in candidate_paths:
            candidate_hash = self._calculate_file_hash(candidate_path)
            if candidate_hash == compiled_hash:
                if self.verbose:
                    print(f"  哈希匹配成功: {candidate_path}")
                return candidate_path
        
        if self.verbose:
            print(f"  警告: 未找到哈希匹配的文件")
        return None
    
    def process_info_file(self, input_file: str, output_file: str):
        """
        处理 lcov info 文件
        
        Args:
            input_file: 输入的 info 文件路径
            output_file: 输出的 info 文件路径
        """
        # 构建文件名索引
        self._build_filename_index()
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise ValueError(f"输入文件不存在: {input_file}")
        
        print(f"正在处理: {input_file}")
        
        mapped_count = 0
        unmapped_count = 0
        total_sf_lines = 0
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                # 检查是否是源文件路径行 (SF:)
                if line.startswith('SF:'):
                    total_sf_lines += 1
                    original_path = line[3:].strip()
                    compiled_path = Path(original_path)
                    
                    if self.verbose:
                        print(f"\n处理文件: {original_path}")
                    
                    # 查找匹配的源文件
                    matched_path = self.find_matching_source(compiled_path)
                    
                    if matched_path:
                        # 找到匹配的源文件，替换路径
                        outfile.write(f"SF:{matched_path}\n")
                        mapped_count += 1
                        if not self.verbose:
                            print(f"✓ {compiled_path.name} -> {matched_path}")
                    else:
                        # 未找到匹配，保持原路径
                        outfile.write(line)
                        unmapped_count += 1
                        print(f"✗ 未找到匹配: {original_path}")
                else:
                    # 非 SF: 行，直接写入
                    outfile.write(line)
        
        # 输出统计信息
        print(f"\n处理完成:")
        print(f"  总文件数: {total_sf_lines}")
        print(f"  成功映射: {mapped_count}")
        print(f"  未能映射: {unmapped_count}")
        print(f"  输出文件: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='修正 lcov info 文件中的源文件路径',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s -i coverage.info -o coverage_fixed.info -s /path/to/source
  %(prog)s -i coverage.info -o coverage_fixed.info -s ./src -v
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入的 lcov info 文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出的 lcov info 文件路径'
    )
    
    parser.add_argument(
        '-s', '--source-dir',
        required=True,
        help='真正的源码目录路径'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='输出详细的处理信息'
    )
    
    args = parser.parse_args()
    
    try:
        mapper = LcovPathMapper(args.source_dir, args.verbose)
        mapper.process_info_file(args.input, args.output)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
