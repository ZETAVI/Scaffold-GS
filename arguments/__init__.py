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
# 当一个文件夹被视为 Python 的包（也就是说，它包含多个 Python 模块）时，__init__.py 文件就会被 Python 解释器自动调用，用于初始化这个包。
# 定义了一些类, 用于处理和管理命令行参数

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

# 自定义的参数解析和参数组构建基类
#   1. 通过继承 ParamGroup 基类，可以直接调用基类的构造函数，将参数添加到 ArgumentParser 对象中
#   2. 如果属性名以_开头，那么这个属性将有一个简写形式。例如，如果属性名为_source_path，那么在命令行中，我们可以使用--source_path或者-s来设置它的值
#   3. 如果fill_none参数为True，那么所有的属性值都会被设置为None
#   4. extract方法用于从命令行参数中提取出与当前参数组相关的参数
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        # * add_argument_group() 方法用于创建一个新的参数组,并将参数添加到这个参数组中
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    # 用户获取用户只定义的参数组
    def extract(self, args):
        group = GroupParams()
        # * vars() 函数返回对象object的属性和属性值的字典对象
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                # * setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是类属性，且可以解析属性
                # 等价于 group.'arg[0]' = 'arg[1]'
                setattr(group, arg[0], arg[1])
        return group

# 继承自 ParamGroup 基类, 在添加控制模型的参数后，直接调用基类____
class ModelParams(ParamGroup): 
    """
    针对3DGS这个特定应用的参数设置，包括模型的几何特征、外观特征、优化参数等。
    """
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # 针对大场景数据集，有较大的视距变化，这意味着在大场景数据集中物体的透明度、协方差和颜色都有显著的变化，如果模型能够考虑这些变化，可能会得到更好的渲染效果。
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    # args：并没有特殊含义，只是我们调用的时候输入命令行参数
    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    """
    todo 还不知道是关于什么参数
    可能是对整条训练渲染管线整体的控制
    
    Paras:
        debug: 是否开启debug模式
    """
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    """
    是对模型训练优化过程的具体参数设置（学习率、迭代次数、稀疏度等）。
    """
    def __init__(self, parser):
        self.iterations = 30_000
        # 在原版3DGS中对于高斯点位置的优化学习率会随着优化进度而调整
        # 但这里由于使用了锚点,所以可以不用考虑不同场景或不同迭代次数下的学习率调整
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        # 对于不同的MLP编码网络的学习率也会随着优化进度而调整
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
