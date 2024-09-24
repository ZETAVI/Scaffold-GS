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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    """
    代表具体的三维场景几何有关的类，主要工作:加载和保存场景相机和点云信息,若是新场景则还包括点云的初始化工作
    Represents a scene in the Scaffold-GS project. 

    Attributes:(比较重要的属性)
        gaussians (GaussianModel): The Gaussian model associated with the scene.
        train_cameras (dict): The training cameras for the scene.
        test_cameras (dict): The test cameras for the scene.
        cameras_extent (float): The extent of the cameras in the scene.(未知)
    """

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """
        Initializes a new instance of the Scene class.

        Args:
            args (ModelParams): The model parameters.
            gaussians (GaussianModel): The Gaussian model associated with the scene.
            load_iteration (int, optional): The iteration to load the trained model from. Defaults to None.(加载检查点,若为-1则自动找point_cloud下最大的迭代次数)
            shuffle (bool, optional): Whether to shuffle the cameras. Defaults to True.(随机洗牌)
            resolution_scales (list, optional): The resolution scales for the cameras.(可能对训练图片进行了不同规模的下采样) Defaults to [1.0].
            ply_path (str, optional): The path to the PLY file. Defaults to None.
        """
        self.model_path = args.model_path
        # 表示加载到检查点的迭代次数,即当前迭代次数
        self.loaded_iter = None
        self.gaussians = gaussians

        # 加载检查点(这里的检查点在train.py中出现,并不是使用save_checkpoints保存的pytorch模型检查点,而是自定义的状态保存)
        if load_iteration:
            # 若load_iteration为-1，则自动找point_cloud下最大的迭代次数
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 这是scene关键的属性 最后scene_info里面图片会以不同分辨率尺度来存储
        self.train_cameras = {}
        self.test_cameras = {}

        # 根据场景类型加载场景信息并返回（colmap/Blender）
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 这里的eval参数一直为false吗,那测试集不就没有东西了吗  -- 是的 全部用于训练
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, ply_path=ply_path)
        else:
            assert False, "Could not recognize scene type!"

        # todo 并不知道gaussians集合下的appearance编码有何作用
        self.gaussians.set_appearance(len(scene_info.train_cameras))
        
        # self.loaded_iter 为None时，将ply点云文件拷贝到model_path下，并将相机信息写入到cameras.json文件中
        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            # 将训练图片集和测试图片集合并，全部转化为json格式并保存
            if scene_info.test_cameras:
                # *extend是Python列表的一个方法，接受可迭代对象（如列表、元组、集合等）作为参数，然后将这个可迭代对象中的所有元素添加到列表的末尾
                # 这个方法不会返回新的列表，而是直接修改原来的列表。
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                # 将python的字典对象（or 列表 元组） 序列化为JSON格式，并将结果写入到file文件中
                json.dump(json_cams, file)

        # 随机洗牌,打乱训练图片集和测试图片集
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # cameras_extent表示场景的尺度大小,为了能使用不同尺度的场景,我们先使用scene_info.nerf_normalization正则化,最后取出场景的半径来表示尺度
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # print(f'self.cameras_extent: {self.cameras_extent}')

        # 按不同分辨率来保存训练＆测试图片集
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            # todo 低优先级 若使用了加载检查点,则加载对应的点云和MLP模型
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
        else:
            # 从COLMAP的point cloud文件中初始化高斯点云,并用场景正则化后的半径来反映场景的尺度大小,主要用于调整学习率
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        """
        Saves the scene at the specified iteration.

        Args:
            iteration (int): The iteration to save the scene at.
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        """
        Gets the training cameras for the specified scale.
        返回特定分辨率尺度的训练图片集,self.train_cameras是一个{}，key是分辨率尺度，value是训练图片集

        Args:
            scale (float, optional): The scale of the cameras. Defaults to 1.0.

        Returns:
            list: The list of training cameras.
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        Gets the test cameras for the specified scale.

        Args:
            scale (float, optional): The scale of the cameras. Defaults to 1.0.

        Returns:
            list: The list of test cameras.
        """
        return self.test_cameras[scale]