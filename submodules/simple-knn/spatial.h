/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>

// 它定义了一个函数原型
// 这个函数接受一个 torch::Tensor 类型的参数 points，并返回一个 torch::Tensor 类型的结果
torch::Tensor distCUDA2(const torch::Tensor& points);