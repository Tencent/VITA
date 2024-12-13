#!/bin/bash

set -e
set -x

timestamp=`date +'%Y%m%d_%H%M%S'`

######################################################################
export ROOT_PATH=/data/
export ROOT_PATH_2=/data_2/
export ROOT_PATH_4=/data_4/
export CODE_PATH=${ROOT_PATH_2}/cognitron_vl/

######################################################################

export LCVLM_URL=http://172.17.0.242:5001/api

cd ${LOCAL_CODE_PATH}

curl -X PUT ${LCVLM_URL} \
	-H "Content-Type: application/json" \
	-d '{
		"prompts": ["San Francisco is a"],
		"image_path_list": [],
		"video_path_list": []
        }'

set +x
