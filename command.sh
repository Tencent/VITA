########## demo ##########
#### LLM + trained connector
CUDA_VISIBLE_DEVICES=6,7 python video_audio_demo.py \
    --model_path outputs/bunny_video_audio_0717/llava-s1-pretrain_mlp_video/ \
    --model_base /mnt/shared/data1/lhj/model_weights/Mixtral-8x7B_modVocab/mg2hg \
    --image_path asset/vita_log.png \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_zh \
    --question "请描述这张图片。"

#### trained LLM + trained connector
## text query
#mixtral
CUDA_VISIBLE_DEVICES=0,1 python video_audio_demo.py \
    --model_path /mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio_0823/llava-s2-pretrain_video/0823ckpt5400 \
    --image_path asset/vita_log2.png \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --question "请描述这张图片。" \
#nemo
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio_0917/llava-s2-pretrain_video/checkpoint-3200 \
    --image_path asset/vita_log2.png \
    --model_type nemo \
    --conv_mode nemo \
    --question "请描述这张图片。" \
#qwen
CUDA_VISIBLE_DEVICES=2 python video_audio_demo.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_ovsi \
    --image_path asset/vita_log2.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "请描述这张图片。"
#qwen fo
CUDA_VISIBLE_DEVICES=6 python video_audio_demo_nemo_fo.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_neg/checkpoint-6500 \
    --image_path ../Video-MLLM/icon.png \
    --model_type qwen2p5_fo_instruct \
    --conv_mode qwen2p5_instruct \
    --question "图片中的人穿着什么衣服？"

## audio query
#mixtral
CUDA_VISIBLE_DEVICES=0,1 python video_audio_demo.py \
    --model_path /mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio_0823/llava-s2-pretrain_video/0823ckpt5400 \
    --image_path asset/vita_log2.png \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --audio_path asset/q1.wav
#nemo
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_0917/llava-s2-pretrain_video/checkpoint-3200 \
    --image_path asset/vita_log2.png \
    --model_type nemo \
    --conv_mode nemo \
    --audio_path asset/q1.wav
#qwen
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_neg/1021s3neg_ckpt500 \
    --image_path asset/vita_log2.png \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav
#qwen fo
CUDA_VISIBLE_DEVICES=1 python video_audio_demo_nemo_fo.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_neg/checkpoint-2000 \
    --image_path asset/vita_log2.png \
    --model_type qwen2p5_fo_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav

# vllm accelerate
CUDA_VISIBLE_DEVICES=4,5 python video_audio_demo_vllm.py \
    --model_path /mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio_0823/llava-s2-pretrain_video/0823ckpt5400 \
    --image_path share.jpg \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --question "请描述这张图片。"


#### VLMEvalKit
## judging
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /mnt/cfs2/lhj/model_weights/Qwen2.5-14B-Instruct --server-port 23333
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/Qwen1.5-1.8B-Chat --server-port 23333
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/Qwen1.5-7B-Chat --server-port 23333
CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/vicuna-7b-v1.5 --server-port 23333
CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/qwen2-7b-chat --server-port 23333
CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/internlm2_5-7b-chat --server-port 23333
CUDA_VISIBLE_DEVICES=3 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/internlm2-chat-1_8b --server-port 23333
CUDA_VISIBLE_DEVICES=3,4 lmdeploy serve api_server /mnt/cfs/lhj/model_weights/Mixtral-8x7B_modVocab/mg2hg --server-port 23333

## VITA
CUDA_VISIBLE_DEVICES=1 python run.py --data MMBench_TEST_EN_V11 --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=2 python run.py --data MMBench_TEST_CN_V11 --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=1 python run.py --data MMStar --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=2 python run.py --data MMMU_DEV_VAL --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=3 python run.py --data MathVista_MINI --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=4 python run.py --data HallusionBench --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=5 python run.py --data AI2D_TEST --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=1 python run.py --data MMVet --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=1 python run.py --data OCRBench --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=5 python run.py --data MME --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=6,7 python run.py --data MME-RealWorld --model vita_qwen2 --verbose

CUDA_VISIBLE_DEVICES=3 python run.py --data MathVista_MINI MMStar MMMU_DEV_VAL HallusionBench AI2D_TEST OCRBench MME --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=3 python run.py --data AI2D_TEST OCRBench MME --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=4 python run.py --data MMBench_TEST_EN_V11 --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=5 python run.py --data MMBench_TEST_CN_V11 --model vita_qwen2 --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME --model vita --verbose

## llava
CUDA_VISIBLE_DEVICES=1 python run.py --data MMStar MMMU_DEV_VAL MathVista_MINI --model llava_video_qwen2_7b --verbose
CUDA_VISIBLE_DEVICES=2 python run.py --data MMBench_DEV_EN_V11 --model llava_video_qwen2_7b --verbose
CUDA_VISIBLE_DEVICES=3 python run.py --data MMBench_DEV_CN_V11 --model llava_video_qwen2_7b --verbose
CUDA_VISIBLE_DEVICES=4 python run.py --data HallusionBench OCRBench --model llava_video_qwen2_7b --verbose
CUDA_VISIBLE_DEVICES=5 python run.py --data AI2D_TEST --model llava_video_qwen2_7b --verbose



### videomme
VIDEO_TYPE="s,m,l"
NAMES=(lyd jyg wzh wzz zcy by dyh lfy)
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=6 python yt_video_inference_qa_imgs.py \
        --model-path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_ovsi \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_wo_sub_temp \
        --video_dir /mnt/cfs/lhj/videochat2/videomme/Video-MME | tee logs/infer.log
done

wait

VIDEO_TYPE="s,m,l"
NAMES=(lyd jyg wzh wzz zcy by dyh lfy)
for((i=0; i<${#NAMES[@]}; i++)) 
do
    CUDA_VISIBLE_DEVICES=7 python yt_video_inference_qa_imgs.py \
        --model-path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_ovsi \
        --model_type qwen2p5_instruct \
        --conv_mode qwen2p5_instruct \
        --responsible_man ${NAMES[i]} \
        --video_type $VIDEO_TYPE \
        --output_dir qa_w_sub_temp \
        --video_dir /mnt/cfs/lhj/videochat2/videomme/Video-MME \
        --use_subtitles | tee logs/infer.log
done

python parse_answer.py --video_types "s,m,l" --result_dir qa_wo_sub


### HCMME_v2
CUDA_VISIBLE_DEVICES=0,1 python video_audio_hcmme.py \
    --model_path /mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio_0823/llava-s2-pretrain_video/0823ckpt5400 \
    --video_path bbe247927e699fd2ece9eb61e7c8a369.mov \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --audio_path ./test_dialog_20240806_154218.wav

### negtive batch test
# qwen
CUDA_VISIBLE_DEVICES=1 python video_audio_demo_batch.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_neg \
    --video_path bbe247927e699fd2ece9eb61e7c8a369.mov \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q2.wav
# qwen fo
CUDA_VISIBLE_DEVICES=4 python video_audio_demo_batch_fo.py \
    --model_path /mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021/llava-s3-finetune_task_neg/checkpoint-1500 \
    --video_path bbe247927e699fd2ece9eb61e7c8a369.mov \
    --model_type qwen2p5_fo_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q2.wav

CUDA_VISIBLE_DEVICES=0,1 python video_audio_demo_batch.py \
    --model_path /mnt/cfs/lhj/videomllm_ckpt/outputs/vita_video_audio_0823/llava-s2-pretrain_video/0823ckpt5400 \
    --video_path bbe247927e699fd2ece9eb61e7c8a369.mov \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --audio_path ./test_dialog_20240806_154218.wav

CUDA_VISIBLE_DEVICES=6,7 python video_audio_demo_batch_fewshot.py \
    --model_path /data/haojialin/model_weigths/VITA/0821ckpt2200 \
    --video_path bbe247927e699fd2ece9eb61e7c8a369.mov \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --audio_path ./test_dialog_20240806_154218.wav

CUDA_VISIBLE_DEVICES=6,7 python video_audio_demo_batch_fewshot2.py \
    --model_path /data/haojialin/model_weigths/VITA/0821ckpt2200 \
    --video_path bbe247927e699fd2ece9eb61e7c8a369.mov \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --audio_path ./test_dialog_20240806_154218.wav

CUDA_VISIBLE_DEVICES=6,7 python video_audio_demo.py \
    --model_path /data/haojialin/model_weigths/VITA/0818ckpt2200 \
    --image_path ../Video-MLLM/icon.png \
    --model_type mixtral-8x7b \
    --conv_mode mixtral_two \
    --audio_path ../Video-MLLM/neg_files/audios/true7.wav

#### Training
## single node
export PYTHONPATH=./
OUTPUT_DIR=/mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_debug
bash script/train/pretrain_mlp.sh ${OUTPUT_DIR}
bash script/train/finetune.sh ${OUTPUT_DIR}
bash script/train/pretrain_mlp_nemo.sh ${OUTPUT_DIR}
bash script/train/pretrain_audio_mlp_nemo.sh ${OUTPUT_DIR}
bash script/train/finetune_nemo.sh ${OUTPUT_DIR}
bash script/train/pretrain_mlp_qwen.sh ${OUTPUT_DIR}
bash script/train/pretrain_audio_mlp_qwen.sh ${OUTPUT_DIR}
bash script/train/finetune_qwen.sh ${OUTPUT_DIR}
bash script/train/finetuneTask_qwen.sh ${OUTPUT_DIR}
bash script/train/finetuneTaskNeg_qwen.sh ${OUTPUT_DIR}
bash script/train/finetuneTaskNeg_qwen_fo.sh ${OUTPUT_DIR}

## multi nodes
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_DIR=/mnt/cfs2/lhj/videomllm_ckpt/outputs/vita_video_audio_1021
bash script/train/pretrain_mlp_nodes.sh ${OUTPUT_DIR}
bash script/train/finetune_nodes.sh ${OUTPUT_DIR}
bash script/train/finetuneTask_nodes.sh ${OUTPUT_DIR}
bash script/train/pretrain_mlp_nemo_nodes.sh ${OUTPUT_DIR}
bash script/train/finetune_nemo_nodes.sh ${OUTPUT_DIR}
bash script/train/pretrain_mlp_qwen_nodes.sh ${OUTPUT_DIR}
bash script/train/pretrain_audio_mlp_qwen_nodes.sh ${OUTPUT_DIR}
bash script/train/finetune_qwen_nodes.sh ${OUTPUT_DIR}
bash script/train/finetuneTask_qwen_nodes.sh ${OUTPUT_DIR}
bash script/train/finetuneTaskNeg_qwen_nodes.sh ${OUTPUT_DIR}
bash script/train/finetuneTaskNeg_qwen_fo_nodes.sh ${OUTPUT_DIR}

### loss图
MODEL_NAME=vita_video_audio_1004
cd ${MODEL_NAME}
coscmd download -f yongdongluo/plot_loss.py ./
python plot_loss.py llava-s1-pretrain_audio_mlp/log.txt
python plot_loss.py llava-s1-pretrain_mlp_video/log_node_0.txt
python plot_loss.py llava-s2-pretrain_video/log_node_0.txt
python plot_loss.py llava-s3-finetue/log.txt
mv llava-s1-pretrain_audio_mlp/loss_plot.png llava-s1-pretrain_audio_mlp/loss_1_audio.png 
mv llava-s1-pretrain_mlp_video/loss_plot.png llava-s1-pretrain_mlp_video/loss_1.png 
mv llava-s2-pretrain_video/loss_plot.png llava-s2-pretrain_video/loss_2.png 
mv llava-s3-finetue/loss_plot.png llava-s3-finetue/loss_3.png 
coscmd upload llava-s1-pretrain_audio_mlp/loss_1_audio.png haojialin/loss_${MODEL_NAME}/
coscmd upload llava-s1-pretrain_mlp_video/loss_1.png haojialin/loss_${MODEL_NAME}/
coscmd upload llava-s2-pretrain_video/loss_2.png haojialin/loss_${MODEL_NAME}/
coscmd upload llava-s3-finetue/loss_3.png  haojialin/loss_${MODEL_NAME}/ 

### 转移代码
tar --exclude='VITA/outputs' -cvzf VITA.tar.gz VITA
tar -tvzf VITA.tar.gz
tar -xvzf VITA.tar.gz

### 转移权重
tar --exclude='checkpoint-1000/global_step1000' -czvf checkpoint-1000.tar.gz checkpoint-1000

### 清楚残留进程
ps aux | grep 'python.*train.py' | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep 'python' | grep -v grep | awk '{print $2}' | xargs kill -9





#############
pip3 config set global.index-url  https://mirrors.tencent.com/pypi/simple/  
bash /mnt/cfs/H20/setup.sh
##docker
sudo yum update -y
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker

curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum-config-manager --enable nvidia-container-toolkit-experimental
sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker


docker run \
    -itd \
    --gpus all \
    --privileged --cap-add=IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /data:/data \
    -v /mnt/cfs:/mnt/cfs \
    -v /mnt/cfs2:/mnt/cfs2 \
    --net=host \
    --ipc=host \
    --name=vita vita:cuda12.1-torch2.3.1


sudo docker save -o vita_image.tar vita:cuda12.1-torch2.3.1
sudo docker load -i vita_image.tar

cat /proc/self/cgroup


### tmux
tmux new -s lhj -d
tmux new -s gpu -d
tmux new -s test1 -d
tmux new -s test5 -d
tmux new -s test6 -d
tmux new -s test7 -d
tmux new -s test8 -d
tmux attach-session -t gpu
nvitop --monitor full
cd /mnt/cfs/lhj/codes/VITA;conda activate vita_nemo

tmux detach
tmux kill-session -t session_name
