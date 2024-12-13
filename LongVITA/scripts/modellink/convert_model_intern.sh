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


if false
then
	load_dir="${ROOT_PATH_4}/models/OpenGVLab/InternViT-300M-448px/"
	save_dir="${ROOT_PATH_4}/models/OpenGVLab/InternViT-300M-448px_tp8pp1"
	rm -fr ${save_dir}
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/lcvlm_modellink/clip_converter_intern.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 8
       	# -use-te-layernorm-linear

fi

if true
then
	load_dir="${ROOT_PATH_4}/models/OpenGVLab/InternViT-6B-448px-V1-5/"
	save_dir="${ROOT_PATH_4}/models/OpenGVLab/InternViT-6B-448px-V1-5_tp1pp1"
	rm -fr ${save_dir}
	mkdir -p ${save_dir}
	python3 ${CODE_PATH}/lcvlm_modellink/clip_converter_intern.py --download-root ${load_dir} --output ${save_dir} --tensor-parallel-size 1
       	# -use-te-layernorm-linear

fi

set +x
