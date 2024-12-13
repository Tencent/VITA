#!/bin/bash

set -e
set -x

timestamp=`date +'%Y%m%d_%H%M%S'`

######################################################################
export ROOT_PATH=/data/
export ROOT_PATH_2=/data_2/
export ROOT_PATH_4=/data_4/
export CODE_PATH=${ROOT_PATH_2}/cognitron_vl/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/cognitron_vl/
mkdir -p ${LOCAL_ROOT_PATH}
mkdir -p ${LOCAL_CODE_PATH}

apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/

######################################################################
OUTPUT_DIR=${ROOT_PATH_2}/output/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

######################################################################
LOG=${OUTPUT_DIR}/log.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

######################################################################

export HF_HOME="${ROOT_PATH_2}/data/HF_HOME/"
mkdir -p ${HF_HOME}
export HF_ENDPOINT=https://hf-mirror.com
export LCVLM_URL=http://127.0.0.1:5001/api

######################################################################
cd ${LOCAL_CODE_PATH}
rm -fr datasets
mkdir -p datasets
ln -s ${ROOT_PATH}/data/ datasets/CV
ln -s ${ROOT_PATH}/data/LLM datasets/LLM
ln -s ${ROOT_PATH}/data/LMM datasets/LMM

haystack_dir=datasets/LMM/OpenDataLab___MovieNet/raw/240P/tt1533117
haystack_dir=datasets/LMM/OpenDataLab___MovieNet/raw/240P/tt2109248

needle_dataset=lmms-lab/v_niah_needles

python3 lcvlm_modellink/evaluation_lcvlm_v-niah.py \
	--haystack_dir ${haystack_dir} \
	--needle_dataset ${needle_dataset} \
	--prompt_template qwen2 \
	--max_num_frames 4000 \
	--min_num_frames 200 \
	--frame_interval 200 \
	--depth_interval 0.2 \
	--output_dir ${OUTPUT_DIR} \


set +x
