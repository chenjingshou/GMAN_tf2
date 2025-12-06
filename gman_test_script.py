#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GMAN环境测试脚本
用于验证环境配置是否正确
"""

import sys
import os

def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_python_version():
    """测试Python版本"""
    print_section("Python版本检查")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 7:
        print("✓ Python版本正确 (3.7.x)")
        return True
    else:
        print(f"✗ Python版本不正确，建议使用3.7.x，当前为{version.major}.{version.minor}.{version.micro}")
        return False

def test_packages():
    """测试依赖包"""
    print_section("依赖包检查")
    
    packages = {
        'tensorflow': '1.14.0',
        'numpy': '1.16.4',
        'pandas': '0.24.2',
        'scipy': None,
        'yaml': None,
        'tables': None,
    }
    
    results = {}
    for package, expected_version in packages.items():
        try:
            if package == 'yaml':
                import yaml
                version = yaml.__version__ if hasattr(yaml, '__version__') else 'unknown'
            elif package == 'tables':
                import tables
                version = tables.__version__
            else:
                module = __import__(package)
                version = module.__version__
            
            if expected_version and version != expected_version:
                print(f"⚠ {package}: {version} (建议: {expected_version})")
                results[package] = 'warning'
            else:
                print(f"✓ {package}: {version}")
                results[package] = 'ok'
        except ImportError:
            print(f"✗ {package}: 未安装")
            results[package] = 'error'
    
    return all(v != 'error' for v in results.values())

def test_tensorflow_gpu():
    """测试TensorFlow GPU支持"""
    print_section("GPU支持检查")
    
    try:
        import tensorflow as tf
        
        # 检查GPU是否可用
        gpus = tf.config.list_physical_devices('GPU') if hasattr(tf.config, 'list_physical_devices') else []
        
        if len(gpus) > 0:
            print(f"✓ 检测到 {len(gpus)} 个GPU:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            return True
        else:
            print("⚠ 未检测到GPU，将使用CPU训练（速度较慢）")
            return True
    except Exception as e:
        print(f"✗ GPU检测失败: {str(e)}")
        return False

def test_cuda():
    """测试CUDA"""
    print_section("CUDA检查")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA驱动已安装")
            print("\nGPU信息:")
            print(result.stdout)
            return True
        else:
            print("⚠ nvidia-smi命令失败，可能未安装NVIDIA驱动")
            return False
    except FileNotFoundError:
        print("⚠ 未找到nvidia-smi命令，可能未安装NVIDIA驱动或未配置PATH")
        return False

def test_data_files():
    """测试数据文件"""
    print_section("数据文件检查")
    
    data_files = {
        'data/METR-LA/metr-la.h5': '交通流量数据',
        'data/METR-LA/adj_mx.pkl': '邻接矩阵（可选）',
    }
    
    all_exist = True
    for file_path, description in data_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ {description}: {file_path} ({size:.2f} MB)")
        else:
            print(f"✗ {description}: {file_path} (不存在)")
            all_exist = False
    
    return all_exist

def test_data_loading():
    """测试数据加载"""
    print_section("数据加载测试")
    
    data_path = 'data/METR-LA/metr-la.h5'
    if not os.path.exists(data_path):
        print(f"✗ 数据文件不存在: {data_path}")
        return False
    
    try:
        import pandas as pd
        print(f"正在加载数据: {data_path}")
        df = pd.read_hdf(data_path)
        print(f"✓ 数据加载成功!")
        print(f"  形状: {df.shape}")
        print(f"  时间范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"  传感器数量: {df.shape[1]}")
        print(f"  缺失值: {df.isnull().sum().sum()}")
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {str(e)}")
        return False

def test_directories():
    """测试目录结构"""
    print_section("目录结构检查")
    
    dirs = [
        'data',
        'data/METR-LA',
        'checkpoints',
        'logs',
        'results'
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (不存在)")
            os.makedirs(dir_path, exist_ok=True)
            print(f"  已自动创建: {dir_path}")
    
    return True

def main():
    """主函数"""
    print("\n" + "="*60)
    print("  GMAN环境测试脚本")
    print("="*60)
    
    results = {
        'Python版本': test_python_version(),
        '依赖包': test_packages(),
        'TensorFlow GPU': test_tensorflow_gpu(),
        'CUDA': test_cuda(),
        '数据文件': test_data_files(),
        '目录结构': test_directories(),
    }
    
    # 如果数据文件存在，测试加载
    if results['数据文件']:
        results['数据加载'] = test_data_loading()
    
    # 总结
    print_section("测试总结")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境配置正确，可以开始训练。")
    else:
        print("\n⚠️  部分测试未通过，请检查并修复相关问题。")
        print("提示：如果数据文件测试未通过，请先下载数据集。")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)