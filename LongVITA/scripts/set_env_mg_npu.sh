#set -e
#set -x

######################################################################
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

export COMBINED_ENABLE=1
export MULTI_STREAM_MEMORY_REUSE=1

export HCCL_RDMA_TC=160
export HCCL_RDMA_SL=5
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_TIMEOUT=20
#export HCCL_ALGO="level0:NA;level1:ring"

export INF_NAN_MODE_ENABLE=1

export DISTRIBUTED_BACKEND="hccl"


export ASCEND_LAUNCH_BLOCKING=0
#将Host日志输出到串口,0-关闭/1-开启
export ASCEND_SLOG_PRINT_TO_STDOUT=0
#设置默认日志级别,0-debug/1-info/2-warning/3-error
export ASCEND_GLOBAL_LOG_LEVEL=3
#设置Event日志开启标志,0-关闭/1-开启
export ASCEND_GLOBAL_EVENT_ENABLE=0
#设置是否开启taskque,0-关闭/1-开启
export TASK_QUEUE_ENABLE=1
#设置是否开启PTCopy,0-关闭/1-开启
export PTCOPY_ENABLE=1
#设置是否开启2个非连续combined标志,0-关闭/1-开启
export COMBINED_ENABLE=1
#设置特殊场景是否需要重新编译,不需要修改
export DYNAMIC_OP="ADD#MUL"
#HCCL白名单开关,1-关闭/0-开启
export HCCL_WHITELIST_DISABLE=1
#设置HCCL超时时间
export HCCL_CONNECT_TIMEOUT=7200


export HCCL_WHITELIST_DISABLE=1

######################################################################
#export NCCL_NET=IB

#export NCCL_SOCKET_IFNAME="bond1"
#export GLOO_SOCKET_IFNAME="bond1"
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export GLOO_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export NCCL_IB_DISABLE=1

#export GPU_NUM_PER_NODE=16
#export NODE_NUM=1
#export INDEX=0
#export MASTER_ADDR=127.0.0.1
#export WORLD_SIZE=16

export CUDA_DEVICE_MAX_CONNECTIONS=1
#return 0

######################################################################
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:10240"

#export ASCEND_LAUNCH_BLOCKING=1
#export TASK_QUEUE_ENABLE=0

######################################################################
# MindSpeed
#export MEMORY_FRAGMENTATION=1


pip3 install --no-index --find-links=${ROOT_PATH}/software/ -r requirements_npu.txt
return 0

######################################################################
#ROOT_PATH=/data/

apt-get update
apt-get install -y libaio-dev
apt-get install -y python3-pybind11
apt-get install -y python3-dev

rm -fr /usr/local/python3.7.5/bin/

if [ -f /usr/local/bin/python3.9 ]; then
	pip3.9 uninstall -y deepspeed_npu
	pip3.9 uninstall -y deepspeed
	pip3.9 uninstall -y torch_npu
	pip3.9 uninstall -y torch

	pip3.9 install --no-index --find-links=${ROOT_PATH}/software/ torch==2.1.0 torchvision==0.16.0 torch_npu==2.1.0.post3

	pip3.9 install --no-index --find-links=${ROOT_PATH}/software/ -r requirements.txt
	pip3.9 install ${ROOT_PATH}/software/apex-0.1_ascend-cp39-cp39-linux_x86_64.whl

	#cp ${ROOT_PATH}/cognitron_vl/patch/deepspeed==0.14.2/stage3.py /usr/local/lib/python3.9/site-packages/deepspeed/runtime/zero/stage3.py
	#cp ${ROOT_PATH}/cognitron_vl/patch/transformers==4.40.1/modeling_qwen2.py /usr/local/lib/python3.9/site-packages/transformers/models/qwen2/modeling_qwen2.py

else
	pip3 uninstall -y deepspeed_npu
	pip3 uninstall -y deepspeed
	pip3 uninstall -y torch_npu
	pip3 uninstall -y torch

	pip3 install --no-index --find-links=${ROOT_PATH}/software/ torch==2.1.0 torchvision==0.16.0 torch_npu==2.1.0.post6
	#pip3 install --no-index --find-links=${ROOT_PATH}/software/ torch==2.2.0 torchvision==0.17.0 torch_npu==2.2.0.post2

	pip3 install --no-index --find-links=${ROOT_PATH}/software/ -r requirements.txt
	#pip3 install ${ROOT_PATH}/software/apex-0.1_ascend-cp38-cp38-linux_x86_64.whl

	#cp ${ROOT_PATH}/cognitron_vl/patch/deepspeed==0.14.2/stage3.py /root/miniconda3/envs/torch21_python38/lib/python3.8/site-packages/deepspeed/runtime/zero/stage3.py
	#cp ${ROOT_PATH}/cognitron_vl/patch/transformers==4.40.1/modeling_qwen2.py /root/miniconda3/envs/torch21_python38/lib/python3.8/site-packages/transformers/models/qwen2/modeling_qwen2.py

fi
