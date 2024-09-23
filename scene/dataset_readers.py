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
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict #场景正则化后的信息
    ply_path: str

def getNerfppNorm(cam_info):
    """
    这是NeRF++的相机位置归一化方法，用于补偿恢复相机集合到他们的中心点出，且计算出场景半径

    Args:
        cam_info (list): List of camera information.

    Returns:
        dict: A dictionary containing the translation and radius.
            - translate (numpy.ndarray): The translation vector.
            - radius (float): The radius value.

    """
    def get_center_and_diag(cam_centers):
        """
        计算所有相机的中心和与中心最大的距离（对角线）
        Calculate the center and diagonal of the camera centers.

        Args:
            cam_centers (list): List of camera centers.

        Returns:
            tuple: A tuple containing the center and diagonal.
                - center (numpy.ndarray): The center vector.
                - diagonal (float): The diagonal value.

        """
        # 将多个数组按第二维度拼接堆叠 原本cam_centers是个list，里面是3*1的向量，这里拼接成了3*n的矩阵
        cam_centers = np.hstack(cam_centers)
        # axis 参数指定以哪一个维度计算均值，指定维数后其余设置为1，如下就是计算1*n为单位的均值,也就是按列拆开再合并计算
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        # 用于计算向量（一维数组）和矩阵的范数，范数一般用于衡量向量或矩阵的大小
        # norm() 默认计算二范数，即向量的长度或矩阵元素的平方和的平方根
        # 可以通过 ord 来指定计算的范数类型，L1计算向量中各元素绝对值之和或矩阵的列和的最大值（默认axis=0）
        # axis 标识以N*1的单位来计算二范数，即计算每个相机中心到平均中心的距离
        # 这里是按行拆开,然后再合并计算,也就是计算每个相机离中心的L2范数,得到的一个1*n的向量
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        # 取相机距离最远的作为场景的对角线
        diagonal = np.max(dist)
        # flatten() 是一个 NumPy 数组方法，用于将多维数组转换为一维数组。
        # 在这个代码中，center 是一个 3x1 的 NumPy 数组，表示相机中心的向量。
        # 通过调用 flatten() 方法，我们将其转换为一个一维数组，以便更方便地处理和使用
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        # 坐标系转换的变换矩阵，用于将目前坐标系下的其他点转换到另一个坐标系中
        # 这里利用相机在世界坐标系中的外参（旋转和偏移），并考虑缩放和平移后，可以构造出一个转换到相机坐标系中的变换矩阵
        # 即将其他点以相机为原点做相对变换
        # 这里W2C 和 C2W 是相互逆的
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])    # 前三行，第四列,也就是齐次坐标变换的平移，所以是个3*1的向量

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    # 偏移补偿回去，这样可以取相机中心点为原点
    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    """
    解析colmap相机数据，返回CameraInfo对象列表，记录了相机的内参和在世界坐标系下的外参T和R
    Reads Colmap camera data and returns a list of CameraInfo objects.

    Args:
        cam_extrinsics (dict): A dictionary containing camera extrinsics.
        cam_intrinsics (dict): A dictionary containing camera intrinsics.
        images_folder (str): The path to the folder containing the camera images.

    Returns:
        list: A list of CameraInfo objects.

    Raises:
        AssertionError: If the Colmap camera model is not supported.

    """

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        #* python 标准输出 the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # 利用每个相机外参中的quaternion四元组vector来计算相机的旋转矩阵(3*3)
        R = np.transpose(qvec2rotmat(extr.qvec))
        # 平移向量表示相机在世界坐标系中的位置
        T = np.array(extr.tvec)

        # 根据相机模型来读取焦距 调整相机FOV视野
        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # print(f'FovX: {FovX}, FovY: {FovY}')

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # PIL Image PIL（Python Imaging Library）是Python的一个图像处理库。PIL Image是PIL库中的一个核心类，用于表示一个图像对象。
        # 这个对象提供了许多方法，可以用来获取图像的属性（如尺寸、模式等），以及操作图像（如裁剪、旋转、调整大小、保存等）
        image = Image.open(image_path)  
 
        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    """
        从ply文件中读取点云数据，返回BasicPointCloud点云集对象[位置，颜色，法线]
    """
    plydata = PlyData.read(path)
    # 取出ply文件中的顶点信息，其中plydata并不是一个字典，其内部定义了_element_lookup和_elements两个属性，_element_lookup是一个字典，存储了元素名和元素对象的映射关系，_elements是一个元组，存储了所有的元素对象
    # 在vertices中存储了每个点云的(x,y,z,nx,ny,nz,red,green,blue)信息,即位置，法线，颜色
    vertices = plydata['vertex']
    # 其中vertices['x']是所有点x坐标的一维列表
    # 与hstack相反，vstack表示vertical stack，即垂直堆叠，将多个数组按第一维度拼接堆叠，最后形成np的array即ndarray（N-dimensional array，即N维数组）
    # 最后对ndarray进行转置操作，
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, lod, llffhold=8):
    # 先读取相机的外参和内参
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 读取原始图片
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # * lambda 函数是一种匿名函数，它可以在一行代码中定义简单的函数。lambda 函数的语法如下
    # lambda arguments: expression
    # 在这个特定的 lambda 函数中，x 是函数的参数，表示 cam_infos_unsorted 列表中的每个元素。x.image_name 是 lambda 函数的输出，表示要根据 image_name 属性进行排序
    # sorted() 是 Python 内置的函数之一，接受两个参数：要排序的可迭代对象和一个可选的 key 参数。可迭代对象是指可以按顺序访问其元素的对象，例如列表、元组或字符串。key 参数是一个函数，用于指定排序的依据。
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 根据模式划分训练数据和测试数据,如果设置了lod即从lod位置进行前后划分,否则使用llffhold来跳跃的对数据集划分
    if eval:
        if lod>0:
            print(f'using lod, using eval')
            if lod < 50:
                #  enumerate(可迭代对象) 返回enumerate对象，是一个迭代器，每次迭代返回一个包含两个元素的元组[索引，元素]
                # *列表推导式， [output_expression for element(这里是迭代器的输出，此处是元组) in iterable if condition == True]
                # c for "idx, c" in enumerate(cam_infos) if idx <= lod
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f'test_cam_infos: {len(test_cam_infos)}')
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            # 根据 llffhold 来跳跃的对数据集划分
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # 对场景的尺度正则化
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 读取稀疏点云
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        # 保存点云为ply文件 只在第一次打开场景时运行
        storePly(ply_path, xyz, rgb)
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_debug=False, undistorted=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            
            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            else:
                im_data = np.array(image.convert("RGBA"))
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if ply_path is None:
        ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# import这个datesetreader文件后，就有这个sceneloadtypecallbacks字典，就可以通过关键字来调用函数
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}