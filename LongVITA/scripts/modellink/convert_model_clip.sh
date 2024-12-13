#!/bin/bash

set -e
set -x


export ROOT_PATH=/apdcephfs_cq10/share_2992827/odysseyshen/
export CODE_PATH=${ROOT_PATH}/open_source/cognitron_vl/

export ROOT_PATH=/data/
export ROOT_PATH_2=/data_2/
export ROOT_PATH_4=/data_4/
export CODE_PATH=${ROOT_PATH_2}/cognitron_vl/

Megatron_path=${CODE_PATH}/third_party/Megatron-LM/
#Megatron_path=${CODE_PATH}/third_party/Megatron-LM_core_r0.6.0/

export PYTHONPATH=${Megatron_path}/:${PYTHONPATH}


if true
then
	load_dir="${ROOT_PATH_4}/models/openai/clip-vit-large-patch14-336"
	save_dir="${ROOT_PATH_4}/models/openai/clip-vit-large-patch14-336_tp8pp1"
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/lcvlm_modellink/clip_converter.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 8
       	# -use-te-layernorm-linear
	#python3 ${Megatron_path}/examples/multimodal/clip_converter.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 4 --use-te-layernorm-linear


fi

set +x
