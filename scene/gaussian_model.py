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

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding

    
class GaussianModel:
    """
    A class representing All Gaussians in a specific scene.

    This class encapsulates the functionality of a Gaussian Model, which is used for modeling
    and rendering 3D scenes. It contains methods for setting up the model(初始化), training the model(训练优化),
    capturing and restoring the model's state(加载和保存高斯点状态), and accessing various properties of the model.

    Attributes:
        feat_dim (int): The dimension of the feature space.(特征编码的长度)
        n_offsets (int): The number of offsets.(锚点附带的高斯点数量)
        voxel_size (float): The voxel size.
        update_depth (int): The update depth.
        update_init_factor (int): The update initialization factor.
        update_hierachy_factor (int): The update hierarchy factor.
        use_feat_bank (bool): Flag indicating whether to use a feature bank.(不同分辨率的特征编码)
        appearance_dim (int): The dimension of the appearance vector.(未知,好像一直是-1)
        embedding_appearance (Embedding): The embedding appearance.(未知,不知与feat_dim有什么区别)
        ratio (int): The ratio.
        add_opacity_dist (bool): Flag indicating whether to add opacity distribution.
        add_cov_dist (bool): Flag indicating whether to add covariance distribution.
        add_color_dist (bool): Flag indicating whether to add color distribution.
        以下都是一些需要网络学习的参数
        _anchor (torch.Tensor): The anchor tensor.
        _offset (torch.Tensor): The offset tensor.
        _anchor_feat (torch.Tensor): The anchor feature tensor.
        opacity_accum (torch.Tensor): The opacity accumulation tensor.
        _scaling (torch.Tensor): The scaling tensor.
        _rotation (torch.Tensor): The rotation tensor.
        _opacity (torch.Tensor): The opacity tensor.
        max_radii2D (torch.Tensor): The maximum radii 2D tensor.
        offset_gradient_accum (torch.Tensor): The offset gradient accumulation tensor.
        offset_denom (torch.Tensor): The offset denominator tensor.
        anchor_demon (torch.Tensor): The anchor denominator tensor.
        optimizer (Optimizer): The optimizer.(具体的优化器)
        percent_dense (int): The percentage of density.
        spatial_lr_scale (int): The spatial learning rate scale.
        mlp_feature_bank (nn.Sequential): The feature bank MLP.(混合三种分辨率下的特征编码的MLP)
        mlp_opacity(cov\color) (nn.Sequential): The opacity MLP.(不透明度MLP\协方差\颜色)
    """

    def setup_functions(self):
        """
        配置不同MLP的激活函数
        Configures the activation functions for different components of the Gaussian model.

        This method sets up the activation functions for scaling, covariance, opacity, and rotation
        in the Gaussian model. It also defines a helper function for building covariance matrices
        from scaling and rotation.（解构出旋转的特征向量）

        Returns:
            None
        """
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # 分解出旋转的特征向量
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # todo 并不知道这里的rotation激活是做什么，并没有rotation吧
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 ):

        self.feat_dim = feat_dim    # 32
        self.n_offsets = n_offsets  # 10
        self.voxel_size = voxel_size     # if voxel_size<=0, using 1nn dist(如果size<0,自动选取初始化时点与点距离的中位数作为voxel_size) 初始化的体素大小
        self.update_depth = update_depth    # growing anchor的三种不同分辨率深度
        self.update_init_factor = update_init_factor    # 初始化时体素的大小对应的比例,设置为16,后续按照16为基准来按比例调整多分辨率体素的大小和阈值
        self.update_hierachy_factor = update_hierachy_factor   # 多分辨率体素每一层之间的比例,会影响每一层稠密化的梯度阈值和体素大小,设置为4
        self.use_feat_bank = use_feat_bank  # 是否三种不同颗粒度的锚点特征融合, 来提高view-adaptability 默认为False

        self.appearance_dim = appearance_dim    # 解码锚点特征的MLP中间层隐藏层的dim
        self.embedding_appearance = None    # MLP中间的隐藏层(就一层 用Embedding(num_cameras, self.appearance_dim).cuda()建模)
        self.ratio = ratio  # 用来控制从pcd中初始化高斯点的比率,不是全部点云都转换为高斯点
        # 针对大场景数据集，有较大的视距变化，这意味着在大场景数据集中物体的透明度、协方差和颜色都有显著的变化，如果模型能够考虑这些变化，可能会得到更好的渲染效果。
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        # 这些带有_的变量都是需要网络学习的参数(以锚点为主体)
        # 锚点的位置(N,3)
        self._anchor = torch.empty(0)
        # 锚点下高斯点的偏移位置(N,k_offsets,3)
        self._offset = torch.empty(0)
        # 锚点的特征(N,feat_dim)
        self._anchor_feat = torch.empty(0)
        
        # 在每一次优化中,统计每个锚点下offset不透明度的总和,用于剔除锚点
        self.opacity_accum = torch.empty(0)

        # 锚点对应的属性
        # (N,6) 
        self._scaling = torch.empty(0)
        # (N,4)
        self._rotation = torch.empty(0)
        # (N,1)
        self._opacity = torch.empty(0)
        # (N)
        self.max_radii2D = torch.empty(0)
        
        # 在每一次优化中,统计被激活且参加渲染的高斯点的梯度累积,后续在每100次是用于判断稠密化
        self.offset_gradient_accum = torch.empty(0)
        # 计数器 (N,1)
        self.offset_denom = torch.empty(0)
        # 每次优化中高斯点涉及的像素点
        self.offset_denom_pixels = torch.empty(0)

        # 在每一次优化中,统计被激活参与渲染的次数
        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        # 用于剔除, 场景对角线的1%, 控制高斯点的密度
        self.percent_dense = 0
        self.spatial_lr_scale = 0   # 空间学习率缩放,即colmap空间对角线半径
        self.setup_functions()

        if self.use_feat_bank:
            # 用于混合三种分辨率下的特征编码的MLP,输入是三维的view-direction和一维的distance,输出是三个混合权重
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                # dim 使得输出向量的每个元素都在0到1之间
                nn.Softmax(dim=1)
            ).cuda()

        # 这里针对大场景视距变化较大，多一个视距信息（一般对于室内场景不启用）
        # 这里视距信息对不透明度和颜色解码做了区分,分别由add_opacity_dist和add_color_dist和add_cov_dist控制
        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        # MLP结构为：36->32->ReLU->n_offsets(生成高斯点的数量)->Tanh激活
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        # 建模高斯点的协方差 旋转4和缩放3 后续还要做处理
        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        # MLP结构为: 36->32->ReLU->7*n_offsets(将旋转4和缩放3结合)->后续分别激活处理
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        # ? 负责颜色 不知道这里为什么加上appearance_dim
        self.color_dist_dim = 1 if self.add_color_dist else 0   # 解码颜色时是否考虑视距信息
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()


    def eval(self):
            """
            设置为评估模式:
            Puts the model in evaluation mode by setting all the necessary components to evaluation mode, 
            which disables dropout and batch normalization layers.
            Dropout层会随机将输入张量的一部分元素设置为0，以防止过拟合。但在评估模式下，Dropout层不会改变输入，因为在评估模型时我们希望使用完整的模型，而不是随机丢弃一部分的模型。
            
            """
            self.mlp_opacity.eval()
            self.mlp_cov.eval()
            self.mlp_color.eval()
            if self.appearance_dim > 0:
                self.embedding_appearance.eval()
            if self.use_feat_bank:
                self.mlp_feature_bank.eval()

    def train(self):
        """
        设置为训练模式
        Trains the model by updating the weights of the MLPs and embeddings.

        This method trains the opacity, covariance, color, and appearance MLPs by calling their respective `train` methods.
        If the `appearance_dim` is greater than 0, the embedding appearance is also trained.
        If the `use_feat_bank` flag is set, the feature bank MLP is trained as well.

        Returns:
            None
        """
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        """获取模型的所有状态信息

        Returns:
            Tuple: A tuple containing the following information
        """
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
            """
            Restores恢复 the model's state from the given arguments.

            Args:
                model_args (tuple): A tuple containing the model arguments.
                training_args (object): The training arguments.

            Returns:
                None
            """
            (self.active_sh_degree, 
            self._anchor, 
            self._offset,
            self._local,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            self.training_setup(training_args)
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    # ? 疑问：为什么需要将训练相机个数编码成appearance_dim维度的向量
    # 这里好像不是GLO的思想（因为GLO是每张图片一个embedding，而这里对于整个Gaussian就只有一个embedding）
    def set_appearance(self, num_cameras):
        # appearance_dim = 32
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    # * @property 装饰器可以将一个方法转换为属性，使得我们可以像访问属性一样来调用这个方法，而不需要在调用时添加括号
    # 这种方式的好处是，你可以在访问属性的同时执行一些额外的操作，例如计算或者被访问前的预处理验证等。在这里就需要做激活的计算
    # @property 装饰器创建的属性默认是只读的。如果你想要创建一个可写的属性，你需要使用 @property 的配套装饰器 @<property_name>.setter
    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # render的时候使用
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        """
        一开始对点云体素化处理，计算出锚点的位置
        Voxelize a sample by rounding the data points to the nearest voxel coordinates.

        Args:
            data (ndarray): The input data points.
            voxel_size (float): The size of each voxel.

        Returns:
            ndarray: The voxelized data points.

        """
        # 先打乱点云数据
        np.random.shuffle(data)
        # 再四舍五入迁移到最近的体素坐标上，然后使用unique去重，最后乘上体素大小还原回原来的尺度中
        # * np.unique()函数是去除数组中的重复数字，并进行排序之后输出
        # 在np.unique()函数中，axis参数用于指定沿哪一个轴来对比，在np中axis=0指的是最外一层[]，所以按照axis=0(相当于arr[0]=arr[0,:],往后遍历)来取就能实现按行取元素排序去重的效果
        # 同理，axis=1，即等于arr[:,0]\arr[:,1]....，按列取元素排序去重
        # 综上，这也是np数组中axis（轴）的规律，一般用于取元素，分组，axis=0是最外层[]中取元素
        # *无论是np还是tensor，设定axis=i的含义可以理解为沿着第i个下标变化的方向进行分组计算操作。也即是将矩阵进行分组，然后再操作
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        从pcd(pointcloud data)文件中创建初始化的高斯点
        Create a Gaussian model from a point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.
            spatial_lr_scale (float): 空间学习率尺度,由训练集中场景半径决定(在3DGS中有讨论,由于colmap的特性对于不同的场景最终的场景半径都控制在5附近)

        Returns:
            None
        """
        # 看看spatial_lr_scale对于不同场景是不是都在5附近 好像是
        # 场景的对角线半径
        self.spatial_lr_scale = spatial_lr_scale
        # 这里ratio可以做到跳点采样初始化,以控制高斯点数量
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            # 将二维的points数组转换为浮点形式的tensor 并移动到gpu上
            init_points = torch.tensor(points).float().cuda()
            # 计算点云中每个点到其他点的距离
            init_dist = distCUDA2(init_points).float().cuda()
            # kthvalue 是 PyTorch 中的一个函数，用于在张量中找到第 k 小的值及其索引。它的作用类似于排序，但只返回第 k 小的值，而不是对整个张量进行排序。
            # 这里K=点数的一半,通过这种方式，你可以有效地找到点云中每个点到其他点的距离的中位数，这在某些算法中可能是一个重要的统计量。
            # 使用距离的中位数作为初始体素的大小
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        # 将点云体素化,points[Anchor,3],表示锚点位置
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom_pixels = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 统计一段时间内锚点的梯度信息累积，用于指导锚点的增长
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask, pixels):
        """
        统计一段时间内锚点的梯度信息累积，用于指导锚点的增长
        Args:
            viewspace_point_tensor (torch.Tensor): 在某个iteration真正被激活的offset的所有高斯点的mean2D梯度
            opacity (torch.Tensor): 所有锚点下所有offset高斯点的不透明度(neural_opacity)
            update_filter (torch.Tensor): 所有被激活的offset高斯点的visible mask (即在视野内的高斯点 radii>0)
            offset_selection_mask (torch.Tensor): 每个锚点具体被激活的offset
            anchor_visible_mask (torch.Tensor): 基于体素的锚点可见性
        Updates:
            self.opacity_accum (torch.Tensor): 所有锚点的不透明度累积
            self.anchor_demon (torch.Tensor): 锚点被激活的计数器 Counts the number of times each anchor is visited.
            self.offset_gradient_accum (torch.Tensor): 所有的offset高斯点的梯度累积(只更新被激活且radii>0的高斯点) Accumulates gradient norms for selected offsets.
            self.offset_denom (torch.Tensor): 所有高斯点被激活且radii>0可见的计数器 Counts the number of times each offset is updated.
        """

        # update opacity stats
        # 取出所有高斯点的不透明度
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        # 每个锚点上都是n_offsets个高斯点,所以这里可以对temp_opacity进行整理,用来计算每个锚点上的综合不透明度
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis 计数器
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        # anchor_visible_mask本来形状为(N),这里先扩展为(N,1),然后复制n_offsets份(N,offset)，最后展平,即所有高斯点的mask
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        # 激活的anchor中被激活的offset 高斯点
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        # 上面的步骤都是先确定哪些offset被激活

        # 在激活的anchor点中再次被激活的offset中,并不是所有被激活的高斯点都是可见的,有些被激活的高斯点可能也在视野之外
        # 在cuda渲染的时候传入的是所有offset激活的高斯点,但是只有可见的高斯点才会被渲染
        # 这里combined_mask[temp_mask]先选择出被激活的高斯点(即真正传到cuda考虑渲染的点),但是否真正渲染还要看是否在视野内,所以这里updata_filter和combined_mask[temp_mask]一样形状
        combined_mask[temp_mask] = update_filter
        
        # 只对mean2D XY方向上的梯度做正则化,按行来求二范数
        grad_norm_pixels = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * pixels[update_filter]
        self.offset_gradient_accum[combined_mask] += grad_norm_pixels
        self.offset_denom[combined_mask] += 1
        self.offset_denom_pixels[combined_mask] += pixels[update_filter]

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            # 大16倍的阈值为threhold, 大4倍的阈值为threshold*2, 一样大的阈值为threshold*4
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            # candidata_mask 用来在所有offset中选择出梯度较高的高斯点
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            # 计算出所有offset高斯点基于锚点的偏移位置 
            # self.get_scaling[:,:3].unsqueeze(dim=1) --- (N,1,3)
            # self._offset (N,k_offsets,3)
            # 最终的形状是(N,k_offsets,3)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            # 计算当前体素的大小,即分辨率,update_init_factor=16,update_hierachy_factor=4, 所以三种分辨率有比voxel_size大16倍/大4倍/一样大/还有小4倍的
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            # 计算出所有需要稠密化被选中的offset高斯点在当前voxelsize下的坐标
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            # 其实这个unique相当于排序(但去重结果不一定严格升序,可以通过sorted (bool) 参数指定),然后去重,最后返回去重后结果的值
            # 并返回原来tensor的元素在去重后的位置 (因为这里return_inverse=True)
            #  --- Whether to also return the indices for where elements in the original input ended up in the returned unique list.
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            # 使用chunk将现有的锚点位置分批,分批后再与当前稠密化需要的新锚点位置做匹配
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    # 这里selected_grid_coords_unique和grid_coords的形状是不一样的(n,1,3)与(4096,3),所以要扩展一下
                    # 这里的目的就是找出所有需要被稠密化的offset,在当前voxelsize下,是否能重用一些已有的锚点.(即需要稠密化的offset计算出来需要的锚点位置是否和已有的锚点位置一样)
                    # 所以这里grid_coords就是根据当前已有的锚点和当前的voxelsize计算出来的锚点坐标(结果为n,4096,3)
                    # .all(-1)：沿最后一个维度（即第 3 维）进行逻辑与操作,即需要的锚点中心与已有的锚点位置完全相等(变为n,4096)
                    # .any(-1)：沿最后一个维度（即第 2 维）进行逻辑或操作,即只要需要的锚点中心与已有的匹配(all已经判断为是否完全相等)(变为n)
                    # .view(-1)：将结果展平(其实没有变化,本来就是一维的)
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    # cur_remove_duplicates 是一个一维的tensor,最终都append进remove_duplicates_list中
                    remove_duplicates_list.append(cur_remove_duplicates)

                # 用于将多个布尔张量合并为一个布尔张量。具体来说，它使用逻辑或 (torch.logical_or) 操作将 remove_duplicates_list 中的所有布尔张量逐元素进行或运算
                # 这里remove_duplicates_list中每个tensor的形状都一样,最终得到一个布尔张量 remove_duplicates，表示所有块中重复坐标的合并结果
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            # 考虑去重和复用锚点之后,还需要稠密化的锚点坐标,这里*cur_size是为了从当前voxel的坐标系转换到世界坐标系,其实还是体素的中心
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                # 新锚点的scaling初始化为cur_size
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                # 新锚点的不透明度初始化为0.1
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                # 用来一个比较笨的方法来创建新锚点的特征初始化,将锚点的特征都复制K_offset份,然后通过需要稠密化的offset高斯点的mask来选择(这里的结果是n,32)
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                # scatter_max是另外一个库pytorch_scattor中的函数
                # scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=None)
                # index表示out[index]位置的值,如果发生冲突则选最大值,在dim维度上进行最大值的聚合(即变动dim来移动,这里dim=0,即按行来进行index和src的聚合)
                # index和scr形状要一样,这里new_feat(n,32)还没考虑重复,所以这里用inverse_indices来合并重复锚点的特征,选取32个维度中每个维度的最大值
                # 如new_feat:torch.Size([42124, 32]),index:torch.Size([42124, 32]) 结果为torch.Size([41632, 32]) --- 到这里得到的是去重后需要的锚点的特征
                # 因为考虑已有锚点的复用,所以最后还需要选取出还没有的锚点位置,[remove_duplicates],结果为torch.Size([40110, 32])
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                # offset每个维度上的偏移都初始化为1
                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                
                # 添加新锚点的计数器
                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                

    # 锚点稠密化
    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom_pixels # [N*k, 1]
        grads[grads.isnan()] = 0.0
        # 因为一个锚点下有offset个高斯点,所以这里计算二范数来代表该锚点的区块
        grads_norm = torch.norm(grads, dim=-1)
        # 在K次优化中,某个offset高斯点要至少激活40%*K次才考虑稠密化 --- 表示剔除或稠密化的候选offset高斯点
        # 这里squeeze将竖排变为横排
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        # 因为前面对anchor点进行了增长,所以这里对offset_denom进行扩展,扩充数量为新锚点的数量
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)
        
        self.offset_denom_pixels[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom_pixels.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom_pixels.device)
        self.offset_denom_pixels = torch.cat([self.offset_denom_pixels, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        # # 在K次优化中,某个offset高斯点要至少激活80%*K次才考虑剔除
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        # 前面做了剔除,需要相应减少相关梯度计数器的数量(先对与offset有关的计数器，后续还会对基于锚点的计数器)
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom
       
        offset_denom_pixels = self.offset_denom_pixels.view([-1, self.n_offsets])[~prune_mask]
        offset_denom_pixels = offset_denom_pixels.view([-1, 1])
        del self.offset_denom_pixels
        self.offset_denom_pixels = offset_denom_pixels

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            # 对于本次不透明度检测的锚点,先对其不透明度累积重置
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        # 再删除掉不透明度较低需要剔除的锚点对应的统计信息空间
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        # 剔除完后重置每个锚点的半径尺度
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
