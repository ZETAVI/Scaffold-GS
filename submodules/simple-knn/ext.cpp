/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 这段代码是在 C++ 中使用 Pybind11 库创建 Python 扩展模块的示例。
 * Pybind11 是一个高级别的接口，用于在 C++11 中创建 Python 的 C++ 扩展和进行 Python 与 C++ 类型之间的转换。
 */

#include <torch/extension.h>
#include "spatial.h"

// PYBIND11_MODULE 是一个宏，用于声明 Python 的扩展模块。
// 它接受两个参数：模块名和一个模块变量。在这个例子中，模块名是 TORCH_EXTENSION_NAME(simple_knn)，模块变量是 m。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def 是一个函数，用于在 Python 模块中定义新的函数。它接受两个参数：函数名和函数的指针。
  // 在这个例子中，函数名是 "distCUDA2"，函数的指针是 &distCUDA2。
  // 这意味着在 Python 中，你可以通过 TORCH_EXTENSION_NAME.distCUDA2 来调用这个 C++ 函数。
  m.def("distCUDA2", &distCUDA2);
}
