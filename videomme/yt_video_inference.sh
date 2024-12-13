VIDEO_TYPE="m,l"
NAMES=(lyd jyg wzh wzz zcy by dyh lfy)
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=4,5 python yt_video_inference_qa.py \
        --model-path /data/haojialin/model_weigths/VITA/checkpoint-3200 \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_wo_sub_ckpt3200 \
        --video_dir /data/haojialin/Video-MME | tee logs/infer.log
done

wait

for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=4,5 python yt_video_inference_qa.py \
        --model-path /data/haojialin/model_weigths/VITA/checkpoint-3200 \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_w_sub_ckpt3200 \
        --video_dir /data/haojialin/Video-MME \
        --use_subtitles | tee logs/infer.log
done
