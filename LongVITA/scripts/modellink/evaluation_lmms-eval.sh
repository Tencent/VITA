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

export HF_HOME="${ROOT_PATH_2}/data/HF_HOME/"
mkdir -p ${HF_HOME}
export HF_ENDPOINT=https://hf-mirror.com


huggingface-cli login --token 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 --add-to-git-credential

######################################################################
LOG=${OUTPUT_DIR}/log.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export ASCEND_PROCESS_LOG_PATH=${OUTPUT_DIR}/ascend/
mkdir -p ${ASCEND_PROCESS_LOG_PATH}

######################################################################

export ASCEND_RT_VISIBLE_DEVICES=0
unset RANK
unset WORLD_SIZE

cd third_party/lmms-eval


######################################################################
# judge / choice extractor
# lmdeploy serve api_server --backend pytorch --device ascend /data_4/models/Qwen/Qwen1.5-1.8B-Chat/ --server-port 23333

export OPENAI_API_KEY=sk-123456
export OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
#export LOCAL_LLM=Qwen/Qwen2-7B-Instruct/
#export LOCAL_LLM=/data_4/models/Qwen/Qwen2-7B-Instruct/
export LOCAL_LLM=/data_4/models/Qwen/Qwen1.5-1.8B-Chat/

export LCVLM_URL=http://127.0.0.1:5001/api
#export LCVLM_URL=http://127.0.0.1:5002/api


	#--tasks=longvideobench_val_i,longvideobench_val_v,longvideobench_test_i,longvideobench_test_v \

python3 -m lmms_eval \
	--model=lcvlm \
	--model_args=max_frames_num=128 \
	--tasks=longvideobench_val_i \
	--batch_size=1 \
	--log_samples \
	--log_samples_suffix=lcvlm \
	--output_path="./logs/" \
	#--wandb_args=project=lmms-eval,job_type=eval,entity=llava-vl

python3 -m lmms_eval \
	--model=lcvlm \
	--model_args=max_frames_num=64 \
	--tasks=longvideobench_val_i \
	--batch_size=1 \
	--log_samples \
	--log_samples_suffix=lcvlm \
	--output_path="./logs/" \
	#--wandb_args=project=lmms-eval,job_type=eval,entity=llava-vl

set +x
