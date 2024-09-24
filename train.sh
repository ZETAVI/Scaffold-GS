# 这个脚本其实就是一个总的训练调用脚本，只不过不像single_train.sh那样这么多可视化的东西,但整体的命令类是

#* shell中定义函数,不需要指定变量,返回值直接echo即可
function rand(){
    min=$1
    #* $((...))是算法运算符，计算并返回结果
    max=$(($2-$min+1))
    num=$(date +%s%N)
    echo $(($num%$max+$min))  
}

#* $() 或者反引号 ` ` 包裹一段命令时,是命令替换语法,会执行命令并将结果替换$()或者` `的位置
#* shell中调用函数只需简单罗列函数名和参数即可，不需要括号
port=$(rand 10000 30000)

lod=0
iterations=30_000
warmup="False"
#* 参数解析，使用了 while 循环和 case 语句来处理传入的命令行参数
# while [[ "$#" -gt 0 ]]; do ... done 是一个 while 循环，它会一直执行，直到没有更多的命令行参数为止。
# * [[ ... ]] 是一个条件表达式，用于进行逻辑判断。它比传统的 [ ... ] 提供了更多的功能，例如字符串的模式匹配和正则表达式
# 这里的 $# 是一个特殊的 Shell 变量，表示当前命令行携带的参数数量。-gt 是一个比较运算符，表示"大于"
# 这里在 $# 上加双引号是出于习惯，因为在其他场景下变量的值可能包含空格或特殊字符（虽然说这里$#一定是数字）
while [[ "$#" -gt 0 ]]; do
    case $1 in
        # `shift` 是一个内置命令，用于后移命令行参数。当你执行 shift 命令时，$1 的值会被丢弃，$2 的值会变成 $1，$3 的值会变成 $2，以此类推。这样，你就可以在一个循环中依次处理所有的命令行参数。
        #  `;;` 是 Shell 脚本中 case 语句的一部分。在 case 语句中，每一个模式后面都需要跟一个 ;;，表示这个模式的结束。
        # 如果当前的值匹配了这个模式，那么就会执行这个模式后面的命令，然后跳过剩下的所有模式，直接到 esac 结束 case 语句
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        --lod) lod="$2"; shift ;;
        --gpu) gpu="$2"; shift ;;
        --warmup) warmup="$2"; shift ;;
        --voxel_size) vsize="$2"; shift ;;
        --update_init_factor) update_init_factor="$2"; shift ;;
        --appearance_dim) appearance_dim="$2"; shift ;;
        --ratio) ratio="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$warmup" = "True" ]; then
    # ${} 用于变量替换。它的作用是获取变量的值。不同于$()是命令替换
    # python train.py --eval -s data/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
    python train.py --eval -s /usr/data/home/lzb/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --warmup --iterations ${iterations} --port $port -m outputs/${data}/${logdir}_warmup/$time
else
    # python train.py --eval -s /usr/data/home/${data}/${logdir} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
    python train.py --eval -s /usr/data/home/lzb/${data} --lod ${lod} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --iterations ${iterations} --port $port -m outputs/${data}/${logdir}/$time
fi