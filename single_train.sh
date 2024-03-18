scene='mipnerf360/garden'
exp_name='baseline'
# voxel_size：对SfM点进行体素化的大小，较小的值表示更精细的结构和更高的开销，
# “0”表示使用每个点的1-NN距离的中值作为体素大小。
voxel_size=0.001
# update_init_factor：增长新锚点的初始分辨率。较大的锚点将开始以较粗的分辨率放置新的锚点。
update_init_factor=16
appearance_dim=0
ratio=1
# 指定使用的gpu，-1表示最空闲的gpu
gpu=3

# example:
./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio}