scene='lzb/classicColmap'
# logdir：训练保存路径
exp_name='02_apple2'
# voxel_size：对SfM点进行体素化的大小，较小的值表示更精细的结构和更高的开销，(即空间划分的更细)
# “0”表示使用每个点的1-NN距离的中值作为体素大小。(统计每个点到最近邻点距离的中值来作为体素大小，即会根据输入数据的密度自动调整)
voxel_size=0.001
# update_init_factor：增长新锚点的初始分辨率。较大的锚点将开始以较粗的分辨率放置新的锚点(即特征长度较短)。
update_init_factor=16
appearance_dim=0
ratio=1
# 指定使用的gpu，-1表示最空闲的gpu
gpu=0

# example:
./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}