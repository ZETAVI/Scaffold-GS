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

import os
import numpy as np

import subprocess
# -q 参数表示查询模式，-d(display) Memory 参数表示只查询内存相关的信息。
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# 创建一个子程序来运行命令，shell=True表示在shell中执行命令，stdout=subprocess.PIPE表示将命令的输出重定向到一个管道中
# 将子进程的输出从字节流解码为字符串。stdout表示子进程的标准输出，.decode()将字节流解码为字符串
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# [:-1] 是一个切片操作，表示除了最后一个元素的所有元素。（因为上面的split('\n')会多出一个空字符串）
# x.split()[2] 将每一行按照空格分割，并取出第 3 个元素，这个元素是 GPU 的内存使用量。int() 函数将这个元素转换为整数。
# np.argmin() 函数找出列表中最小元素的索引，也就是内存使用最少的 GPU 的编号
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# os.environ['CUDA_VISIBLE_DEVICES']="1"

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
#import _init_中的方法和gaussian_render模块中的其他方法
from gaussian_renderer import prefilter_voxel, render, network_gui 
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    """
    Copy the runtime code to the specified destination directory, excluding files and directories specified in .gitignore.

    Args:
        dst (str): The destination directory where the runtime code will be copied to.

    Returns:
        None
    """
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    # Read .gitignore file
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        # *当你对一个文件对象使用 for 循环时，Python 会默认使用逐行读取的方式。这是因为文件对象是一个可迭代对象，其迭代器的 __next__() 方法被设计为每次返回文件的下一行。
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    # 将 ignorePatterns 从集合（set）转换为列表（list）的主要原因是因为后续的 shutil.ignore_patterns(*ignorePatterns) 函数需要一个可迭代的参数，而且这个参数通常是一个列表或元组
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    # __file__ 是一个内置的 Python 变量，它表示当前正在执行的脚本文件的路径。
    # pathlib.Path(__file__) 使用 pathlib 模块的 Path 类将 __file__ 转换为一个 Path 对象。Path 对象提供了一些方法和属性，可以方便地操作和获取路径的各个部分
    # .parent 属性获取了这个 Path 对象的父目录。这个父目录也是一个 Path 对象
    # .resolve() 方法将这个 Path 对象转换为一个绝对路径
    log_dir = pathlib.Path(__file__).parent.resolve()

    # shutil.copytree(src：源目录的路径, dst：目标目录的路径,ignore: 这个可调用对象应返回一个相对于 src 的、不应被复制的文件名列表) 函数用于递归地复制一个目录树。这个函数的目标目录不能已经存在。
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    """

    Args:
        dataset (arguments.GroupParams Object): 针对3DGS具体实例的一些参数
        opt (arguments.GroupParams): 训练相关的参数
        pipe (arguments.GroupParams): 对训练渲染管线某些环节的控制
        dataset_name (str): 实例名称
        testing_iterations (list): 评估的迭代次数
        saving_iterations (list): 保存的迭代次数
        checkpoint_iterations (list): 默认空
        checkpoint (_type_): None
        debug_from (int): 默认-1
        wandb (_type_, optional): _description_. Defaults to None.
        logger (_type_, optional): _description_. Defaults to None.
        ply_path (_type_, optional): _description_. Defaults to None.
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    # 实例化高斯点集，用于维护高斯点的稠密化和优化
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    # 实例化场景，主要于三维场景几何有光，负责维护相机位置和高斯点
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    # 针对于高斯训练的一些参数（学习率等）
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 创建了一个Pytorch CUDA事件，用于同步GPU操作，通过设置两个事件的记录和同步，可以计算两个事件之间的时间差
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # 通过迭代次数设置进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # 开始训练
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # # network gui not available in scaffold-gs yet 用于显示的网络连接
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     # 这一部分场景渲染并传输展示部分,要在训练结束后才进行
        #     try:
        #         net_image_bytes = None
        #         # 从网络gui接收用户端数据(如相机位置\训练控制等)
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        # 根据训练的阶段来调整学习率
        gaussians.update_learning_rate(iteration)

        # 设置背景颜色,并转化为tensor
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        # Pick a random Camera,从训练图像集中随机选取一个视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render & evaluate 从debug_from开始调试(由于设置为-1，所以只有在最后进行一次质量评估)
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 计算锚点可视化的范围
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        # 是否继续更新高斯点，即是否继续计算梯度
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # 渲染 得到渲染结果包
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, pixelgs_depth_scaled_gard=opt.pixelgs_depth_threshold*scene.cameras_extent)

        # visibility_filter是所有被激活anchor下激活的offset在cuda渲染时的可见性, visible_mask是锚点的可见性(还不是锚点内特定offset的激活)
        # radii与scaling有关,但radii是2D投影后scaling的长边的1.5倍(考虑正太分布的三个标准差内 99.73%)
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        # .cuda() 是 PyTorch 中的一个方法，用于将 Tensor 对象移动到 GPU 上进行计算。
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        # 对高斯的尺度正则化，取当前高斯点集中每个高斯点尺度(按行)的乘积的均值
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        # 自动计算损失函数关于模型参数的梯度,这些梯度会被存储在对应的参数.grad属性中
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar 代码更新了一个名为ema_loss_for_log的指标，这是一个指数移动平均损失，用于平滑损失的变化并在进度条中显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                # 设置进度条的后缀信息，主要是更新一个键为 "Loss"的字典项值，设置为保留7为小数格式化后的损失值(注意值是str格式"{}",{}是占位符),:.{7}f为格式化
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                # tpdm库的进度条更新，每次增加10,默认加1,也可以加小数
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # densification 在500 - 15000 迭代之间累积各锚点的梯度，并从1500开始每ope.update_interval对锚点进行稠密化
            if iteration < opt.update_until and iteration > opt.start_stat:
                # 用来对每次优化迭代的梯度做累积的,其中包括了更新次数的记录
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, render_pkg["pixels"])

                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                # 清空cuda梯度的缓存
                torch.cuda.empty_cache()

            # Optimizer step 
            #! 优化器是一个对象，其作用是使用backword的梯度来更新模型参数，以最小化损失函数，当你调用optimizer.step()时，优化器会根据其内部的优化算法（如Adam、SGD等）来更新模型的参数。
            # 优化器需要知道哪些Tensor是模型的参数，因此在创建优化器时，我们需要将模型的参数（这些参数也是Tensor）传递给优化器。（也就是说tensor有多个，但是优化器可以只有一个）
            #! 形成了深度学习中的“前向传播 - 计算损失 - 反向传播 - 更新参数”的基本流程
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                # 将所有优化器的梯度置空，优化器的梯度和Tensor的梯度虽然说都是更新模型参数的值，但是Tensor的梯度是直接存储在Tensor对象中，而优化器则是使用这些梯度来执行更新操作。这里只是将优化器的梯度置空
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                #  Saves an object to a disk file.
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, opt, pipe):
    """
    Prepares the output folder and create logger for the model.

    Args:
        args: An object containing the command line arguments.

    Returns:
        tb_writer: A Tensorboard writer object if Tensorboard is available, otherwise None.
    """
    if not args.model_path:
        # 对于没有指定模型输出路径的情况，我们会在当前目录下创建一个名为output的文件夹，并在其中创建一个唯一的子文件夹，用于存储模型的输出
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    # * 第二种格式化字符串方法
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "cfg_args_origin"), 'w') as cfg_log_f:
        cfg_log_f.write(' '.join(sys.argv))

    with open(os.path.join(args.model_path, "cfg_args_opt"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(opt))))

    with open(os.path.join(args.model_path, "cfg_args_pipe"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(pipe))))


    #  # 每次运行备份文件
    # if os.path.exists(args.model_path + "/backup"):
    #     count = 1
    #     for entry in os.listdir(args.model_path):
    #         entry_path = os.path.join(args.model_path, entry)
    #         if os.path.isdir(entry_path) and entry.startswith('backup'):
    #             count += 1
    #     cmd = "rsync -a --exclude='outputs' --exclude='submodules/diff-gaussian-rasterization/cuda_rasterizer/stdOut' --exclude='submodules/diff-gaussian-rasterization/build' . " + args.model_path + "/backup"  + str(count)
    #     os.system(cmd)
    # else:
    #     cmd = "rsync -a --exclude='outputs' --exclude='submodules/diff-gaussian-rasterization/cuda_rasterizer/stdOut' --exclude='submodules/diff-gaussian-rasterization/build' . " + args.model_path + "/backup"
    #     os.system(cmd)


    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    """
    在训练过程中Generates a training report for the given iteration.

    Parameters:
    - tb_writer (SummaryWriter): TensorBoard summary writer for logging.
    - dataset_name (str): Name of the dataset.
    - iteration (int): Current iteration number.
    - Ll1 (torch.Tensor): L1 loss value.
    - loss (torch.Tensor): Total loss value.
    - l1_loss (torch.nn.Module): L1 loss function.
    - elapsed (float): Time elapsed for the iteration.（运行时间）
    - testing_iterations (list): List of iteration numbers for testing.
    - scene (Scene): Scene object containing scene information.
    - renderFunc (function): Rendering function.
    - renderArgs (tuple): Arguments for the rendering function.
    - wandb (wandb.Run, optional): WandB run object for logging. Default is None.
    - logger (logging.Logger, optional): Logger object for logging. Default is None.

    Returns:
    - None
    """
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    error_image = error_map(image, gt_image)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/errormap2".format(viewpoint.image_name), error_image[None], global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()



                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))


                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize();t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)

            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

def get_logger(path):
    """
    Create and configure a logger object，并创建两个日志处理器，分别做日志的写入和输出

    Args:
        path (str): The path to the directory where the log file will be saved.

    Returns:
        logging.Logger: The configured logger object.

    """
    import logging

    logger = logging.getLogger()
    # 这意味着所有级别为INFO及以上的日志都会被记录
    logger.setLevel(logging.INFO)
    # 创建FileHandler日志处理器，用于将日志写入文件
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    # 同样是日志处理器，但是这个处理器用于将日志输出到控制台
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    # * ArgumentParser 是一个类，用于解析命令行参数。它的构造函数有很多参数，其中最重要的一个是 description，用于描述这个命令行程序的功能。
    # * parser所包含的命令解析，记录在parser._actions列表中，每个命令解析都是一个Action类的实例，我们还可以为action分类，类别记录在parser._action_groups中（默认已经有positional arguments和optional arguments两个类别）
    # 具体见ModelParams(针对3DGS的参数), PipelineParams, OptimizationParams(训练相关的参数)是如何添加解析Action和分类的
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # nargs="+" 表示这个选项的值可以是一个或多个参数。例如，用户可以在命令行中输入 --test_iterations 10000 20000 30000，那么 args.test_iterations 就会是一个列表 [10000, 20000, 30000]。
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1]+[500*(i) for i in range(61)])
    # save_iterations 是一个列表，里面包含了需要保存模型的时刻。默认情况下，只保存最后一次迭代的模型。
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000*(i) for i in range(61)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')

    #* sys.argv 是一个包含命令行参数的列表。sys.argv[0] 是脚本名（即程序的名字），sys.argv[1:] 是脚本后面跟着的参数
    args = parser.parse_args(sys.argv[1:])
    # 将 args.iterations 的值添加到 args.save_iterations 列表的末尾, iterations 是训练的总迭代次数,可能比save_iterations中的最大值要大
    args.save_iterations.append(args.iterations)


    # enable logging
    # 该变量由-m/-model_path参数指定，用于指定模型的输出路径
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    # * f-string（格式化字符串字面量），你可以在字符串前面添加f或F，然后在字符串中使用花括号{}来包含变量或表达式
    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')



    try:
        # *os.path.join() 方法用于将多个路径组合成一个路径（用/号连接）。如果其中一个路径是绝对路径，则之前的路径将被丢弃，只使用绝对路径。
        # 如果最后一个部分是空字符串，那么结果路径将以路径分隔符结束，即print(os.path.join("/path/to/dir", ""))  # 输出：/path/to/dir/
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG) 设置一些随机数种子，并启动安静模式
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # 设置检测异常的模式，如果设置为True，那么当出现异常时，PyTorch会自动停止程序并打印出异常的位置
    # 例如在反向传播过程中检测到NaN或无穷大值。启用异常检测模式后，一旦检测到这些问题，程序会立即停止，并提供有关错误发生位置的详细信息，这有助于定位问题的源头。
    # 原理是利用了上下文管理器with, 一进入with先检查异常，然后执行with内的代码，如果with内的代码出现异常，就会抛出异常，然后执行with外的代码(类似于try)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
