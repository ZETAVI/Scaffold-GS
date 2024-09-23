#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 这个 setup.py 文件是用来构建和安装 Python 扩展模块的。
# 在这个文件中，我们使用 setuptools 库的 setup 函数来配置我们的扩展模块。

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

# setup() 函数用于配置和安装 Python 包。
# simple_knn 和 simple_knn._C 的关系是包含和被包含的关系。simple_knn 是一个 Python 包，simple_knn._C 是这个包中的一个扩展模块。
# from simple_knn import _C 也可以 import simple_knn._C
setup(
    name="simple_knn",
    ext_modules=[
        # CUDAExtension 是 torch.utils.cpp_extension 中的一个类，它用于创建一个 CUDA 扩展模块。
        # 我们创建了一个名为 simple_knn._C 的 CUDA 扩展模块，它包含了三个源文件：spatial.cu，simple_knn.cu 和 ext.cpp
        # extra_compile_args 参数用于指定额外的编译器参数。在这个例子中，我们为 nvcc（CUDA 编译器）和 cxx（C++ 编译器）指定了不同的参数。
        # 对于 cxx，我们添加了一些编译器标志，这些标志是在之前的代码中根据操作系统类型确定的。
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
