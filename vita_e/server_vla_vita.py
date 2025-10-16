from __future__ import print_function

import argparse
import asyncio
import base64
import builtins
import datetime
import io
import json
import multiprocessing
import os
import sys
import re
import threading
import time
import warnings
from queue import Empty
from typing import AsyncGenerator
from threading import Timer
from collections import deque
import cv2
import pickle
import numpy as np
from PIL import Image
import traceback

from flask import Flask, current_app, render_template, request
from flask_socketio import SocketIO, disconnect, emit

# repo_root = os.path.abspath(os.path.dirname(__file__))
# isaac_root = os.path.join(repo_root, "..", "Isaac-GR00T")
# print(f"Import gr00t from {isaac_root}")
# if os.path.isdir(os.path.join(isaac_root, "gr00t")):
#     sys.path.insert(0, isaac_root)
# else:
#     raise ValueError("Isaac-GR00T not found")

vl_load_root = os.path.join("gr00t", "model", "vl_load")
vl_load_vita_dir = os.path.join(vl_load_root, "vita")
print(f"Import vita from {vl_load_root}")
if os.path.isdir(vl_load_vita_dir):
    sys.path.insert(0, vl_load_root)
else:
    raise ValueError("vita not found")

from vita_e.vita_vla_html.web.parms import GlobalParams
from vita_e.vita_vla_html.web.pem import generate_self_signed_cert
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy_vita_action_head import Gr00tActionHeadPolicy

# 导入工具模块
from vita_e.server_utils_vita import (
    Colors, ProcessConfig, 
    IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX, IMAGE_TOKEN, AUDIO_TOKEN, VIDEO_TOKEN,
    AUDIO_RESPONSE_TOKEN, TEXT_RESPONSE_TOKEN, ACT_TOKEN, HALT_TOKEN, 
    INSTRUCTION_TOKEN, END_TOKEN, SPECIAL_TOKEN_LIST,
    decoder_topk, codec_padding_size, target_sample_rate,
    load_model_embemding, save_video, tokenizer_image_audio_token, 
    clear_queue, merge_current_and_history
)

def get_args():
    parser = argparse.ArgumentParser(description='VITA')
    parser.add_argument('--model_path_vlm', help='VLA model path to load', default='checkpoints/vita_vla_finetune')
    parser.add_argument(
        '--model_path_policy',
        help='GR00T model path for action generation',
        default='checkpoints/vita_gr00t_robot_head',
    )
    parser.add_argument('--data_config_name', help='Data config name', default='real_data_robot_vita_action_head')
    parser.add_argument('--ip', help='ip of server', default='0.0.0.0')
    parser.add_argument('--port', help='port of server', default=8081)
    parser.add_argument('--max_users', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    print(f"{Colors.CYAN}VITA server args: {args}{Colors.RESET}")
    return args

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

args = get_args()
last_tts_model_id = 0

# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print

# init flask app
app = Flask(__name__, template_folder='./vita_vla_html/web/resources', static_folder='./vita_vla_html/web/static')
socketio = SocketIO(app)
# init connected users
connected_users = {}

def disconnect_user(sid):
    if sid in connected_users:
        print(f"{Colors.RED}Disconnecting user {sid} due to time out{Colors.RESET}")
        socketio.emit('out_time', to=sid) 
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]

def load_model(
        llm_id,
        model_path,
        cuda_devices,
        inputs_queue,
        outputs_queue,
        tts_outputs_queue,
        action_feature_queue,
        action_feature_stack,
        stop_event,
        other_stop_event,
        worker_ready,
        wait_workers_ready,
        start_event,
        other_start_event,
        start_event_lock,
        is_generating_action_feature,
        other_is_generating_action_feature,
        current_dialog,
        global_history,
        shutdown_event,
        global_history_limit=0,
        gr00t_model_path=None,
        data_config_name=None,
        observation_queue=None,
        observation_queue_lock=None,
    ):
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    # 导入依赖CUDA的包
    import torch
    import torchaudio
    from transformers import AutoTokenizer
    from decord import VideoReader, cpu
    import torchvision.transforms.v2 as Tv2
    from torchvision.transforms import InterpolationMode
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    from vita.conversation import conv_templates, SeparatorStyle
    from vita.util.mm_utils import (
        KeywordsStoppingCriteria,
        tokenizer_image_audio_token as vita_tokenizer_image_audio_token,
        tokenizer_image_token as vita_tokenizer_image_token,
    )
    from vita.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN, IMAGE_TOKEN_INDEX as VITA_IMAGE_TOKEN_INDEX
    
    #等待tts初始化
    print(f"{Colors.MAGENTA}wait_workers_ready: {wait_workers_ready}{Colors.RESET}")
    wait_workers_ready[1].wait()
    print(f"{Colors.MAGENTA}wait_workers_ready status updated: {wait_workers_ready}{Colors.RESET}")
    print("model_path", model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载原生 VITA 模型与处理器
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    vita_model = VITAQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
    vita_model.to(device=device, dtype=torch.bfloat16)
    vita_model.resize_token_embeddings(len(tokenizer))
    vision_tower = vita_model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor
    audio_encoder = vita_model.get_audio_encoder()
    audio_encoder.to(device=device, dtype=torch.bfloat16)
    audio_processor = audio_encoder.audio_processor
    
    # 预计算用于动作结束判定的token id序列
    action_end_token_ids = tokenizer.encode(END_TOKEN, add_special_tokens=False)
    print(f"{Colors.MAGENTA}END token id heads: {action_end_token_ids[:1]}{Colors.RESET}")

    # 初始化GR00T action policy
    print(f"{Colors.CYAN}Initializing GR00T action policy from {gr00t_model_path}{Colors.RESET}")
    data_config = DATA_CONFIG_MAP[data_config_name]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    gr00t_action_policy = Gr00tActionHeadPolicy(
        model_path=gr00t_model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag="new_embodiment",
        denoising_steps=4,
    )
    print(f"{Colors.CYAN}GR00T action policy initialized successfully{Colors.RESET}")

    def _process_inputs(inputs):

        def _process_image(image_path):
            if isinstance(image_path, str):
                assert os.path.exists(image_path), f"Image file {image_path} does not exist."
                return Image.open(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            else:
                assert isinstance(image_path, np.ndarray), "Image must be either a file path or a numpy array."
                return Image.fromarray(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)


        def _process_audio(audio_path):
            # 为与原生 VITA 一致的音频管线，这里保留原始路径，后续用 audio_processor.process 构建 audios 字典
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
            return audio_path
        
        def _process_video(video_path, max_frames=4, min_frames=4, s=None, e=None):
            # speed up video decode via decord.

            if s is None or e is None:
                start_time, end_time = None, None
            else:
                start_time = int(s)
                end_time = int(e)
                start_time = max(start_time, 0)
                end_time = max(end_time, 0)
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = start_time + 1

            if os.path.exists(video_path):
                vreader = VideoReader(video_path, ctx=cpu(0))
            else:
                raise FileNotFoundError(f"Video file {video_path} does not exist.")

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            
            if num_frames > 0:
                # T x 3 x H x W
                all_pos = list(range(f_start, f_end + 1))
                if len(all_pos) > max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
                elif len(all_pos) < min_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
                else:
                    sample_pos = all_pos

                # patch_images = [
                #     Image.fromarray(f).transpose(Image.FLIP_LEFT_RIGHT)
                #     for f in vreader.get_batch(sample_pos).asnumpy()
                # ]
                patch_images = [
                    Image.fromarray(f)
                    for f in vreader.get_batch(sample_pos).asnumpy()
                ]
                return patch_images

            else:
                print(f"{Colors.RED}video path: {video_path} error.{Colors.RESET}")

        def _process_image_multi_modal_data(inputs):
            """处理图像模态数据并验证token数量"""
            if "image" not in inputs.get("multi_modal_data", {}):
                return
            
            image_inputs = inputs["multi_modal_data"]["image"]
            if not isinstance(image_inputs, list):
                image_inputs = [image_inputs]
            inputs["multi_modal_data"]["image"] = [_process_image(f) for f in image_inputs]

            if "prompt" in inputs:
                assert inputs["prompt"].count(IMAGE_TOKEN) == len(image_inputs), \
                    f"Number of image token {IMAGE_TOKEN} in prompt must match the number of image inputs."
            elif "prompt_token_ids" in inputs:
                num_image_tokens = inputs["prompt_token_ids"].count(IMAGE_TOKEN_INDEX)
                assert num_image_tokens == len(image_inputs), (
                    f"Number of image token ids {IMAGE_TOKEN_INDEX} in "
                    f"prompt_token_ids must match the number of image inputs."
                )
            else:
                raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

        def _process_audio_multi_modal_data(inputs):
            """处理音频模态数据并验证token数量"""
            if "audio" not in inputs.get("multi_modal_data", {}):
                return
            
            audio_inputs = inputs["multi_modal_data"]["audio"]
            if not isinstance(audio_inputs, list):
                audio_inputs = [audio_inputs]
            inputs["multi_modal_data"]["audio"] = [_process_audio(f) for f in audio_inputs]

            if "prompt" in inputs:
                num_audio_inputs = len(inputs["multi_modal_data"]["audio"])
                assert inputs["prompt"].count(AUDIO_TOKEN) == num_audio_inputs, (
                    f"Number of audio token {AUDIO_TOKEN} in prompt must "
                    f"match the number of audio inputs."
                )
            elif "prompt_token_ids" in inputs:
                num_audio_inputs = len(inputs["multi_modal_data"]["audio"])
                num_audio_tokens = inputs["prompt_token_ids"].count(AUDIO_TOKEN_INDEX)
                assert num_audio_tokens == num_audio_inputs, (
                    f"Number of audio token ids {AUDIO_TOKEN_INDEX} in "
                    f"prompt_token_ids must match the number of audio inputs."
                )
            else:
                raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

        def _process_video_multi_modal_data(inputs):
            """处理视频模态数据并验证token数量"""
            if "video" not in inputs.get("multi_modal_data", {}):
                return
            
            video_inputs = inputs["multi_modal_data"]["video"]
            if not isinstance(video_inputs, list):
                video_inputs = [video_inputs]

            assert "prompt" in inputs, (
                "Prompt must be provided when video inputs are provided."
            )
            assert "image" not in inputs["multi_modal_data"], (
                "Image inputs are not supported when video inputs are provided."
            )
            assert inputs["prompt"].count(VIDEO_TOKEN) == 1, (
                "Currently only one video token is supported in prompt."
            )
            num_video_inputs = len(inputs["multi_modal_data"]["video"])
            assert inputs["prompt"].count(VIDEO_TOKEN) == num_video_inputs, (
                f"Number of video token {VIDEO_TOKEN} in prompt must match "
                f"the number of video inputs."
            )
            
            video_frames_inputs = []
            for video_input in video_inputs:
                video_frames_inputs.extend(_process_video(video_input, max_frames=4, min_frames=4))
            num_video_frames = len(video_frames_inputs)
            inputs["prompt"] = inputs["prompt"].replace(
                VIDEO_TOKEN, IMAGE_TOKEN * num_video_frames
            )
            if "image" not in inputs["multi_modal_data"]:
                inputs["multi_modal_data"]["image"] = []
            inputs["multi_modal_data"]["image"].extend(video_frames_inputs)

            inputs["multi_modal_data"].pop("video", None)

        # 处理多模态数据
        if "multi_modal_data" in inputs:
            _process_image_multi_modal_data(inputs)
            _process_audio_multi_modal_data(inputs)
            _process_video_multi_modal_data(inputs)

        # 构建视觉与音频张量
        all_visual_tensor = []
        has_image_data = (
            "multi_modal_data" in inputs and
            "image" in inputs["multi_modal_data"]
        )
        if has_image_data:
            for img in inputs["multi_modal_data"]["image"]:
                proc = image_processor.preprocess(img, return_tensors="pt")
                all_visual_tensor.append(proc["pixel_values"][0])
        if len(all_visual_tensor) > 0:
            images_tensor = torch.stack(all_visual_tensor, dim=0)
            images_tensor = images_tensor.to(dtype=vita_model.dtype)
        else:
            images_tensor = torch.zeros((1, 3, 448, 448))
            images_tensor = images_tensor.to(dtype=vita_model.dtype)

        has_audio_data = (
            "multi_modal_data" in inputs and
            "audio" in inputs["multi_modal_data"] and
            len(inputs["multi_modal_data"]["audio"]) > 0
        )
        if has_audio_data:
            latest_audio_path = inputs["multi_modal_data"]["audio"][-1]
            audio_feat, audio_for_llm_len = audio_processor.process(os.path.join(latest_audio_path))
            audio_len = audio_feat.shape[0]
            audios = {
                "audios": audio_feat.unsqueeze(0).to(dtype=torch.bfloat16),
                "lengths": torch.tensor(audio_len).unsqueeze(0).to(dtype=torch.bfloat16),
                "lengths_for_llm": torch.tensor(audio_for_llm_len).unsqueeze(0),
            }
        else:
            audio = torch.zeros(400, 80)
            audio_length = torch.tensor(audio.shape[0])
            audio_for_llm_lens = torch.tensor(60)
            audios = {
                "audios": audio.unsqueeze(0).to(dtype=torch.bfloat16),
                "lengths": audio_length.unsqueeze(0).to(dtype=torch.bfloat16),
                "lengths_for_llm": audio_for_llm_lens.unsqueeze(0),
            }

        return inputs, images_tensor, audios

    def _process_vla_inputs(frame_rgb: np.ndarray, instruction: str) -> torch.Tensor:
        def _apply_eval_image_transform(
            img_rgb: np.ndarray,
            scale: float = 0.95,
            target_size: tuple[int, int] = (224, 224),
        ) -> np.ndarray:
            if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
                return img_rgb
            h, w, _ = img_rgb.shape
            crop_h = max(1, int(h * scale))
            crop_w = max(1, int(w * scale))
            frames_tensor = torch.from_numpy(img_rgb).to(torch.float32) / 255.0
            frames_tensor = frames_tensor.permute(2, 0, 1)
            transform = Tv2.Compose([
                Tv2.CenterCrop((crop_h, crop_w)),
                Tv2.Resize(
                    (target_size[1], target_size[0]),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
            ])
            out_tensor = transform(frames_tensor)
            out_np = (out_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            return out_np

        frame_224 = _apply_eval_image_transform(frame_rgb, scale=0.95, target_size=(224, 224))
        pil_224 = Image.fromarray(frame_224)
        pil_448 = pil_224.resize((448, 448))
        pixel_values = image_processor(images=pil_448, return_tensors="pt")["pixel_values"][0]
        pixel_values = pixel_values.unsqueeze(0).to(dtype=vita_model.dtype, device=device)

        question_prompt = (
            "These two images are views of the same robotic arm from the front and its "
            "end effector position. "
            "Play the role of the robot arm in the picture. Based on the given task "
            "instructions, analyze the color and "
            "shape of the objects in front of you, and understand the relative "
            "position between the end effector of the robot arm and these objects. "
            "Provide as much information as possible to complete the task. Ignore "
            "objects that are not relevant to the task. Task instructions: "
        )
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question_prompt + instruction
        conv = conv_templates["qwen2p5_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("image")
        input_ids = vita_tokenizer_image_token(prompt, tokenizer, VITA_IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device=device)

        # 静音音频占位
        audio = torch.zeros(400, 80, device=device)
        audio_length = torch.tensor(audio.shape[0], device=device)
        audio_for_llm_lens = torch.tensor(60, device=device)
        audios = {
            "audios": audio.unsqueeze(0).to(dtype=torch.bfloat16),
            "lengths": audio_length.unsqueeze(0).to(dtype=torch.bfloat16),
            "lengths_for_llm": audio_for_llm_lens.unsqueeze(0),
        }

        return {"input_ids": input_ids, "pixel_values": pixel_values, "audios": audios}

    def judge_negative(text):
        is_negative = text.startswith('☟')
        return is_negative
    
    def get_first_special_token(text):
        """获取文本开头的特殊token"""
        text = text.strip()
        for token in SPECIAL_TOKEN_LIST:
            if text.startswith(token):
                return token
        return None
    
    def extract_instruction(text):
        """从文本中提取[INSTRUCTION]后面的内容"""
        if INSTRUCTION_TOKEN in text:
            parts = text.split(INSTRUCTION_TOKEN)
            if len(parts) > 1:
                return parts[-1].strip()
        return None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    # 局部变量用于记录上一次的对话和视频帧
    last_dialog, last_video_frame_timestamp = None, 0.0
    action_generation_start_time = None
    action_generation_timeout = 30  # seconds
    last_stack_len = 0

    flag = False
    while not shutdown_event.is_set():
        time.sleep(0.01)
        
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        if not flag:
            print(f"{Colors.CYAN}Process {cuda_devices} is ready.{Colors.RESET}")
            flag = True

        # 语音生成阶段
        if not inputs_queue.empty() and not is_generating_action_feature.is_set():
            # 有输入时，且没有在生成动作特征
            with start_event_lock:
                if start_event.is_set():
                    # start_event为True，表示允许生成
                    inputs = inputs_queue.get()
                    if not other_is_generating_action_feature.is_set():
                        # 另一个模型没有在生成动作特征，则下次让对方生成，自己不生成
                        other_start_event.set()
                        start_event.clear()
                else:
                    # 自己不被允许生成
                    continue
            
            # 合并历史构建输入
            current_request = inputs.copy()
            inputs = merge_current_and_history(
                global_history[-global_history_limit:],
                inputs,
                skip_history_vision=True,
                skip_history_audio=True,
                move_image_token_to_start=True
            )

            # 基于合并后的 inputs 构建视觉与音频特征
            print(f"{Colors.YELLOW}inputs: {inputs}{Colors.RESET}")
            inputs, all_visual_tensor, audios = _process_inputs(inputs)
            all_visual_tensor = all_visual_tensor.to(device=device)
            audios["audios"] = audios["audios"].to(device=device)
            audios["lengths"] = audios["lengths"].to(device=device)
            audios["lengths_for_llm"] = audios["lengths_for_llm"].to(device=device)

            assert "prompt" in inputs, "Prompt must be provided after merging history."
            input_ids = (
                vita_tokenizer_image_audio_token(
                    inputs["prompt"],
                    tokenizer,
                    VITA_IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(device=all_visual_tensor.device)
            )

            # 生成输出
            llm_start_time = time.time()
            print(f"{Colors.YELLOW}input_ids device: {input_ids.device}{Colors.RESET}")
            print(f"{Colors.YELLOW}images device: {all_visual_tensor.device}{Colors.RESET}")
            print(f"{Colors.YELLOW}audios device: {audios['audios'].device}{Colors.RESET}")

            stop_str = "<|im_end|>"
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = vita_model.generate(
                    input_ids,
                    images=all_visual_tensor,
                    audios=audios,
                    do_sample=False,
                    temperature=0.01,
                    top_p=None,
                    num_beams=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=512,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    shared_v_pid_stride=None,
                )
            llm_end_time = time.time()
            print(f"{Colors.GREEN}LLM process time: {llm_end_time - llm_start_time}{Colors.RESET}")

            seq_ids = output_ids.sequences
            llm_output = tokenizer.batch_decode(seq_ids, skip_special_tokens=False)[0]
            llm_output = llm_output.strip()
            if llm_output.endswith(stop_str):
                llm_output = llm_output[:-len(stop_str)]
            llm_output = llm_output.strip()
            print(f"{Colors.GREEN}LLM output: {llm_output}{Colors.RESET}")
            
            # 获取首个特殊token
            first_token = get_first_special_token(llm_output)
            print(f"{Colors.GREEN}First special token: {first_token}{Colors.RESET}")
            
            # 处理首token逻辑
            is_first_time_to_work = True
            if is_first_time_to_work:  # 检查第一个token的情况
                stop_event.clear()  # 允许把自己的输出发给下游，不被打断
                if not other_is_generating_action_feature.is_set():
                    other_stop_event.set()  # 如果对方没有在生成动作特征，可能在输出语音，则打断对方（语音打断）
                
                if first_token == HALT_TOKEN:
                    other_stop_event.set()  # 如果首 token 是急停，则打断对方（急停打断）
                    other_is_generating_action_feature.clear()  # 停止对方的动作特征生成
                    action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})  # 输出急停信号
                    
                elif first_token == ACT_TOKEN:
                    # 如果首 token 是要输出动作，则开始生成动作特征
                    # 若对方未处于生成阶段，先清空对方未完成的动作，否则之后回撤对方动作
                    if not other_is_generating_action_feature.is_set():
                        while len(action_feature_stack) > 0:
                            action_feature_stack.pop() # 清空栈
                    # 如果对方在生成动作特征，则打断对方（动作打断）
                    else:
                        print(
                            f"{Colors.RED}LLM {llm_id} is interrupting other action feature "
                            f"generation.{Colors.RESET}"
                        )
                        other_stop_event.set()  # 如果对方在生成动作特征，则打断对方（动作打断）
                        other_is_generating_action_feature.clear()
                        other_start_event.set()  # 打断对方以后，下次让对方生成，自己不生成，自己进入动作特征生成阶段
                        start_event.clear()  # 自己下次不被允许生成
                    
                    is_generating_action_feature.set()  # 如果首 token 是要输出动作，则开始生成动作特征
                    action_generation_start_time = time.time() # 开始计时
                    # 通知前端或客户端开始更新机器人状态
                    print(
                        f"{Colors.YELLOW}LLM {llm_id} is notifying frontend to start "
                        f"updating robot states.{Colors.RESET}"
                    )
                    action_feature_queue.put({"id": llm_id, "action": "START_ACTION"})
                    try:
                        instruction_content = extract_instruction(llm_output)
                        if instruction_content:
                            current_dialog.value = instruction_content  # 获取动作指令，并赋值给共享变量
                        else:
                            warnings.warn("No [INSTRUCTION] found in outputs when [ACT] is found")
                            current_dialog.value = current_request.get("prompt", "")
                    except Exception as e:
                        warnings.warn(f"Error extracting instruction: {e}")
                        current_dialog.value = current_request.get("prompt", "")
                
                clear_queue(outputs_queue)  # 清空输出队列，准备输出自己的 token
                clear_queue(tts_outputs_queue)
                is_first_time_to_work = False

            # 处理输出到TTS队列（除了[INSTRUCTION]后面的部分）
            if not stop_event.is_set():
                # 分割输出，只发送[INSTRUCTION]之前的部分到TTS
                if INSTRUCTION_TOKEN in llm_output:
                    tts_output = llm_output.split(INSTRUCTION_TOKEN)[0]
                else:
                    tts_output = llm_output
                
                # 添加首句标记并发送到TTS
                tts_output = '$$FIRST_SENTENCE_MARK$$' + tts_output
                
                # 按字符处理TTS输出
                history_generated_text = ''
                for char in tts_output:
                    if stop_event.is_set():
                        print(f"{Colors.RED}LLM {llm_id} is interrupted.{Colors.RESET}")
                        break
                    
                    history_generated_text += char
                    history_generated_text = history_generated_text.replace('☞ ', '').replace('☞', '')
                    
                    if char in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"]:
                        outputs_queue.put({"id": llm_id, "response": history_generated_text})
                        history_generated_text = ''

            # 保存对话历史（仅原始请求的可序列化副本）
            current_request["response"] = llm_output
            if llm_output.strip():
                global_history.append(current_request)

        # 动作特征生成阶段
        elif is_generating_action_feature.is_set():
            # 每轮先检查是否被打断
            if stop_event.is_set():  # [TAG1] 检查是否被打断
                print(f"{Colors.RED}LLM {llm_id} is interrupted.{Colors.RESET}")
                is_generating_action_feature.clear()  # 如果被打断，则停止生成动作特征
                action_generation_start_time = None
                # 恢复到可接收下一条语音输入的状态
                stop_event.clear()
                other_stop_event.clear()
                with start_event_lock:
                    start_event.set()
                    other_start_event.clear()
                action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})  # 输出急停
                continue

            if action_generation_start_time and (
                time.time() - action_generation_start_time > action_generation_timeout
            ):
                print(
                    f"{Colors.RED}LLM {llm_id} action generation timed out after "
                    f"{action_generation_timeout} seconds.{Colors.RESET}"
                )
                is_generating_action_feature.clear()
                action_generation_start_time = None
                stop_event.clear()
                other_stop_event.clear()
                with start_event_lock:
                    start_event.set()
                    other_start_event.clear()
                action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN}) # 发送停止信号
                continue
            
            if is_generating_action_feature.is_set():  # 如果还在生成动作特征，没有被打断
                # 检查是否有新的对话或视频帧
                current_dialog_text = current_dialog.get("value", "")
                # current_dialog_text = "Pick up the red toy and place into the basket."
                current_observation = None
                with observation_queue_lock:
                    if not observation_queue.empty():
                        current_observation = observation_queue.get()
                    else:
                        continue

                if current_observation is not None:
                    # 前端每次发送观测时，最多只回撤一次（与生成新动作频率对齐）
                    # 回撤逻辑移动到获取观测之后：仅当栈顶是对方且本轮观测尚未回撤时，执行一次回撤并跳过本轮动作生成
                    stack_len = len(action_feature_stack)
                    if stack_len > 0:
                        top_action_data = action_feature_stack[-1]
                        top_action_id = top_action_data.get("id", None)
                        if stack_len != last_stack_len:
                            print(
                                f"{Colors.YELLOW}LLM {llm_id} is checking action feature stack: "
                                f"{stack_len}{Colors.RESET}"
                            )
                            print(
                                f"{Colors.YELLOW}LLM {llm_id} is checking top action data: "
                                f"{top_action_id}{Colors.RESET}"
                            )
                            last_stack_len = stack_len
                        if isinstance(top_action_data, dict) and top_action_id != llm_id:
                            # 仅回撤一次，并在本轮观测内不再生成新动作
                            retraction_action_data = action_feature_stack.pop()
                            retraction_action = retraction_action_data.get("action", retraction_action_data)
                            action_feature_queue.put({"id": llm_id, "action": retraction_action, "type": "retraction"})
                            print(
                                f"{Colors.YELLOW}Executing retraction action from llm "
                                f"{retraction_action_data.get('id', 'unknown')}.{Colors.RESET}"
                            )
                            continue

                    # 构建动作特征生成的输入
                    print(f"{Colors.YELLOW}LLM {llm_id} is generating action feature.{Colors.RESET}")
                    frame_data = current_observation.get("data", None)
                    robot_states = current_observation.get("states", {})  # 机器人状态

                    video_frame_array = None
                    if frame_data is not None:
                        video_frame_array = pickle.loads(frame_data)
                    
                    if video_frame_array is not None:
                        vla_inputs = _process_vla_inputs(video_frame_array, current_dialog_text)
                        # 前向，提取最后一层 hidden states
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = vita_model(
                                input_ids=vla_inputs["input_ids"],
                                images=vla_inputs["pixel_values"],
                                audios=vla_inputs["audios"],
                                output_hidden_states=True,
                                return_dict=True,
                                use_cache=True,
                            )
                        
                        # 检查是否应该结束动作生成：基于下一token预测判定 ACTION_END
                        next_token_logits = outputs.logits[:, -1, :]  # (1, vocab)
                        next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
                        # 兼容解码判定，避免BPE未收录导致的漏判
                        predicted_token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)

                        # if (next_token_id == action_end_token_ids[0]) or (
                        #     END_TOKEN in predicted_token_str
                        # ):
                        #     print(
                        #         f"{Colors.RED}LLM {llm_id} action generation ended by END token "
                        #         f"(id={next_token_id}).{Colors.RESET}"
                        #     )
                        #     is_generating_action_feature.clear()
                        #     action_generation_start_time = None
                        #     stop_event.clear()
                        #     other_stop_event.clear()
                        #     with start_event_lock:
                        #         start_event.set()
                        #         other_start_event.clear()
                        #     action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})
                        #     continue

                        hidden_states = outputs.hidden_states[-1].squeeze(0)
                        print(
                            f"{Colors.YELLOW}Successfully extracted hidden states: "
                            f"{hidden_states.shape}{Colors.RESET}"
                        )
                        
                        # 使用GR00T action policy生成动作
                        robot_observations = {
                            "state.hand": np.array(robot_states.get("hand"), dtype=np.float32)[None, :],
                            "state.robot": np.array(robot_states.get("robot"), dtype=np.float32)[None, :],
                        }
                        action_dict = gr00t_action_policy.get_action(hidden_states, observations=robot_observations)
                        
                        # 将CUDA Tensor转换为Python列表以便跨进程传递和JSON序列化
                        serializable_action_dict = {}
                        for k, v in action_dict.items():
                            if isinstance(v, torch.Tensor):
                                serializable_action_dict[k] = v.cpu().numpy().tolist()
                            elif isinstance(v, np.ndarray):
                                serializable_action_dict[k] = v.tolist()
                            else:
                                serializable_action_dict[k] = v
                        
                        # 将动作特征和owner_id一起压入栈，并发送到队列
                        action_feature_stack.append({"id": llm_id, "action": serializable_action_dict})
                        action_feature_queue.put({"id": llm_id, "action": serializable_action_dict, "type": "new"})
                        print(
                            f"{Colors.YELLOW}Put action into action_feature_queue: "
                            f"{list(action_dict.keys())}{Colors.RESET}"
                        )
                    else:
                        warnings.warn("No video frame found, cannot generate action")
                        is_generating_action_feature.clear()
                        action_generation_start_time = None
                        stop_event.clear()
                        with start_event_lock:
                            start_event.set()
                            other_start_event.clear()
            else:  # 如果被打断，则输出急停
                action_feature_queue.put({"id": llm_id, "action": HALT_TOKEN})

def tts_worker(
    model_path,
    inputs_queue,
    outputs_queue,
    worker_ready,
    wait_workers_ready,
    shutdown_event,
):
    # 导入依赖CUDA的包
    import torch
    import torchaudio
    from vita.model.vita_tts.decoder.llm2tts import llm2TTS
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    from transformers import AutoTokenizer

    def remove_uncommon_punctuation(text):
        common_punctuation = ".,!?;:()[]，。！？、：；（） "
        uncommon_punctuation_pattern = rf"[^\w\s{re.escape(common_punctuation)}]"
        cleaned_text = re.sub(uncommon_punctuation_pattern, "", text)

        return cleaned_text
    
    def remove_special_tokens(input_str):
        # Remove special tokens
        special_tokens = ['☞', '☟', '☜', '☝', '☀', '☯', '<unk>', '<|im_end|>']
        for token in special_tokens:
            input_str = input_str.replace(token, '')
        return input_str

    def replace_equation(sentence):

        special_notations = {
            "sin": " sine ",
            "cos": " cosine ",
            "tan": " tangent ",
            "cot": " cotangent ",
            "sec": " secant ",
            "csc": " cosecant ",
            "log": " logarithm ",
            "exp": "e^",
            "sqrt": "根号 ",
            "abs": "绝对值 ",
        }
        
        special_operators = {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            "=": "等于",
            '!=': '不等于',
            '>': '大于',
            '<': '小于',
            '>=': '大于等于',
            '<=': '小于等于',
        }

        greek_letters = {
            "α": "alpha ",
            "β": "beta ",
            "γ": "gamma ",
            "δ": "delta ",
            "ε": "epsilon ",
            "ζ": "zeta ",
            "η": "eta ",
            "θ": "theta ",
            "ι": "iota ",
            "κ": "kappa ",
            "λ": "lambda ",
            "μ": "mu ",
            "ν": "nu ",
            "ξ": "xi ",
            "ο": "omicron ",
            "π": "派 ",
            "ρ": "rho ",
            "σ": "sigma ",
            "τ": "tau ",
            "υ": "upsilon ",
            "φ": "phi ",
            "χ": "chi ",
            "ψ": "psi ",
            "ω": "omega "
        }

        sentence = sentence.replace('**', ' ')

        sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

        for key in special_notations:
            sentence = sentence.replace(key, special_notations[key]) 
        for key in special_operators:
            sentence = sentence.replace(key, special_operators[key])
        for key in greek_letters:
            sentence = sentence.replace(key, greek_letters[key])


        sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
        sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
        
        return sentence

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_embedding = load_model_embemding(model_path).to(device)
    tts = llm2TTS(os.path.join(model_path, 'vita_tts_ckpt'))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    past_llm_id = 0

    while not shutdown_event.is_set():
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        tts_input_text = ""
        while not inputs_queue.empty():
            time.sleep(0.03)

            stop_at_punc_or_len = False
            response = inputs_queue.get()
            llm_id, newly_generated_text = response["id"], response["response"]

            for character in newly_generated_text:
                
                if  past_llm_id != 0 and past_llm_id != llm_id:
                    tts_input_text = ""
                    outputs_queue.put({"id": llm_id, "response": ("|PAUSE|", None, 0.2)})
                
                tts_input_text += character

                past_llm_id = llm_id
                if character in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"] and len(tts_input_text) >= 5:
                    stop_at_punc_or_len = True
                    break

            if stop_at_punc_or_len:
                break

        if tts_input_text.strip() == "":
            continue

        if '$$FIRST_SENTENCE_MARK$$' in  tts_input_text.strip():
            codec_chunk_size = 20
            seg_threshold = 0.1
            tts_input_text = tts_input_text.replace('$$FIRST_SENTENCE_MARK$$', '').replace('，', '。').replace(',', '。')
            IS_FIRST_SENTENCE = True
        else:
            codec_chunk_size = 40
            seg_threshold = 0.015
            IS_FIRST_SENTENCE = False
        tts_input_text = remove_special_tokens(tts_input_text)
        tts_input_text = replace_equation(tts_input_text)
        tts_input_text = tts_input_text.lower()

        if tts_input_text.strip() == "":
            continue
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = llm_embedding(torch.tensor(tokenizer.encode(tts_input_text)).to(device))
        for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                            None, 
                            codec_chunk_size=codec_chunk_size,
                            codec_padding_size=codec_padding_size,
                            seg_threshold=seg_threshold):

            if IS_FIRST_SENTENCE:
                try:
                    split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                    seg = seg[:, :, split_idx:]
                except:
                    print(f'{Colors.MAGENTA}Do not need to split{Colors.RESET}')
                    pass

            seg = torch.cat([seg], -1).float().cpu()
            audio_data = (seg.squeeze().numpy() * 32768.0).astype(np.int16)

            audio_duration = seg.shape[-1]/24000
            if past_llm_id == 0 or past_llm_id == llm_id:
                outputs_queue.put({"id": llm_id, "response": (tts_input_text, audio_data, audio_duration)})

def send_pcm(sid, request_inputs_queue):
    """
    Sends PCM audio data to the dialogue system for processing.
    """
    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    print(f"Sid: {sid} Start listening")
    while True:
        if connected_users[sid][1].stop_pcm:
            print(f"Sid: {sid} Stop pcm")
            connected_users[sid][1].stop_generate = True 
            connected_users[sid][1].stop_tts = True
            break
            
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_size)
        if e is None:
            continue

        res = connected_users[sid][1].wakeup_and_vad.predict(e)

        if res is not None:
            if 'start' in res:
                print(f"Sid: {sid} Vad start")

            elif 'cache_dialog' in res:
                print(f"Sid: {sid} Vad end")

                directory = './chat_history'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                audio_duration = len(res["cache_dialog"]) / target_sample_rate

                if audio_duration < 1:
                    print(f"{Colors.YELLOW}The duration of the audio is less than 1s, skipping...{Colors.RESET}")
                    continue

                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                audio_filename = f"{directory}/test_dialog_{timestamp}.wav"
                torchaudio.save(audio_filename, res["cache_dialog"].unsqueeze(0), target_sample_rate)

                video_filename = None
                if len(connected_users[sid][1].collected_images) > 0:
                    video_filename = f"{directory}/test_video_{timestamp}.mp4"
                    save_video(connected_users[sid][1].collected_images, video_filename)

                print(f"{Colors.BLUE}Start to generate response{Colors.RESET}")
                if video_filename:
                    current_request = {
                        "prompt": "<video><audio>",
                        "multi_modal_data": {
                            "video": [video_filename],
                            "audio": [audio_filename],
                        },
                    }
                else:
                    current_request = {
                        "prompt": "<audio>",
                        "multi_modal_data": {
                            "audio": [audio_filename],
                        },
                    }
                print(f"{Colors.BLUE}Start to put request into queue {current_request}{Colors.RESET}")
                request_inputs_queue.put(current_request)

@app.route('/')
def index():
    return render_template('demo.html')

@socketio.on('connect')
def handle_connect():
    if len(connected_users) >= args.max_users:
        print(f'{Colors.YELLOW}Too many users connected, disconnecting new user{Colors.RESET}')
        emit('too_many_users')
        return

    sid = request.sid
    connected_users[sid] = []
    connected_users[sid].append(Timer(args.timeout, disconnect_user, [sid]))
    connected_users[sid].append(GlobalParams())
    connected_users[sid][0].start()
    
    request_queue = current_app.config['REQUEST_QUEUE']
    pcm_thread = threading.Thread(target=send_pcm, args=(sid, request_queue,))
    pcm_thread.start()
    print(f'{Colors.CYAN}User {sid} connected{Colors.RESET}')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]
    print(f'{Colors.CYAN}User {sid} disconnected{Colors.RESET}')

@socketio.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording stopped')

@socketio.on('audio')
def handle_audio(data):
    global last_tts_model_id
    sid = request.sid
    if sid in connected_users:
        try:
            # 处理action_feature队列中的数据
            action_feature_queue = current_app.config.get('ACTION_FEATURE_QUEUE')
            if action_feature_queue and not action_feature_queue.empty():
                try:
                    action_data = action_feature_queue.get_nowait()
                    action = action_data.get('action')
                    print(f"{Colors.BLUE}Processing action queue: {action_data.keys()}{Colors.RESET}")

                    if action == "START_ACTION":
                        print(
                            f"{Colors.YELLOW}Received START_ACTION signal, emitting "
                            f"start_update_states.{Colors.RESET}"
                        )
                        emit('start_update_states', True)
                    elif isinstance(action, dict):
                        # This is an actual action feature
                        print(f"{Colors.YELLOW}Sending action feature to frontend: {list(action.keys())}{Colors.RESET}")
                        emit('action_feature', action_data)
                        print(f"{Colors.YELLOW}Sending finished.{Colors.RESET}")
                    elif action == HALT_TOKEN:
                        print(f"{Colors.RED}Sending HALT signal to frontend.{Colors.RESET}")
                        emit('action_feature', {"halt": True})
                except Empty:
                    pass
            
            if not current_app.config['TTS_OUTPUT_QUEUE'].empty():
                connected_users[sid][0].cancel()
                connected_users[sid][0] = Timer(args.timeout, disconnect_user, [sid])
                connected_users[sid][0].start()

                tts_output_queue = current_app.config['TTS_OUTPUT_QUEUE']
                try:
                    output_data = tts_output_queue.get_nowait()
                    print(f"{Colors.MAGENTA}output_data: {output_data}{Colors.RESET}")

                    if output_data is not None:
                        llm_id = output_data["id"]
                        _, audio, length = output_data["response"]

                        print(f"{Colors.MAGENTA}llm_id: {llm_id}, last_tts_model_id: {last_tts_model_id}{Colors.RESET}")
                        if last_tts_model_id != llm_id:
                            print(
                                f"{Colors.YELLOW}Received output from other process {llm_id}, "
                                f"last output tts model is {last_tts_model_id}, skipping..."
                                f"{Colors.RESET}"
                            )
                            socketio.emit('stop_tts', to=sid)
                        else:
                            print(f"Sid: {sid} Send TTS data")
                            emit('audio', audio.tobytes())

                        last_tts_model_id = llm_id
                except Empty:
                    pass
        
            if connected_users[sid][1].tts_over_time > 0:
                socketio.emit('stop_tts', to=sid)
                connected_users[sid][1].tts_over_time = 0
            
            data = json.loads(data)
            audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
            sample_rate = data['sample_rate']
            
            connected_users[sid][1].pcm_fifo_queue.put(torch.tensor(audio_data, dtype=torch.float32) / 32768.0)

        except Exception as e:
            traceback.print_exc()
            print(f"{Colors.RED}Error processing audio: {e}{Colors.RESET}")
    else:
        disconnect()

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    if sid in connected_users:
        try:
            image_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            if current_time - connected_users[sid][1].last_image_time > 1:
                connected_users[sid][1].collected_images.clear()
                print(f"{Colors.MAGENTA}Clearing the collected images{Colors.RESET}")
            
            connected_users[sid][1].collected_images.append(frame)
            connected_users[sid][1].last_image_time = current_time
            
        except Exception as e:
            print(f"{Colors.RED}Error processing video frame: {e}{Colors.RESET}")
    else:
        disconnect()

@socketio.on('reset_state')
def handle_reset_state():
    global_history = current_app.config['GLOBAL_HISTORY']
    while len(global_history) > 0:
        global_history.pop()
    print(f"{Colors.CYAN}Resetting the state{Colors.RESET}")

@socketio.on('states')
def handle_states(data):
    observation_queue = current_app.config.get('OBSERVATION_QUEUE')
    observation_queue_lock = current_app.config.get('OBSERVATION_QUEUE_LOCK')
    with observation_queue_lock:
        while not observation_queue.empty():
            observation_queue.get()
        
        # 处理图像数据，与video_frame处理方式统一
        processed_data = data.copy()
        if data.get('data') is not None:
            try:
                # 解码base64图像数据（与handle_video_frame相同的处理方式）
                image_data = base64.b64decode(data['data'].split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用pickle序列化，与原来的处理方式保持一致
                processed_data['data'] = pickle.dumps(frame)
                print(f"{Colors.BLUE}Processed image data, frame shape: {frame.shape}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}Error processing image data: {e}{Colors.RESET}")
                processed_data['data'] = None
        
        # data格式：{"data": pickle.dumps(frame) or None, "states": {"hand": [...], "robot": [...]}}
        observation_queue.put(processed_data)
        print(
            f"{Colors.BLUE}Added robot states to queue: "
            f"hand={len(data.get('states', {}).get('hand', []))}, "
            f"robot={len(data.get('states', {}).get('robot', []))}{Colors.RESET}"
        )

def cleanup_resources():
    """清理多进程资源"""
    print(f"{Colors.CYAN}正在清理资源...{Colors.RESET}")
    with app.app_context():
        # 1. 通知所有子进程关闭
        if 'SHUTDOWN_EVENT' in current_app.config:
            print(f"{Colors.CYAN}发送关闭信号...{Colors.RESET}")
            current_app.config['SHUTDOWN_EVENT'].set()

        # 2. 等待进程结束
        processes_to_join = {
            'MODEL_1_PROCESS': current_app.config.get('MODEL_1_PROCESS'),
            'MODEL_2_PROCESS': current_app.config.get('MODEL_2_PROCESS'),
            'TTS_WORKER_PROCESS': current_app.config.get('TTS_WORKER_PROCESS'),
        }

        for name, process in processes_to_join.items():
            if process and process.is_alive():
                print(f"{Colors.CYAN}等待 {name} 退出...{Colors.RESET}")
                process.join(timeout=10) # 等待10秒

        # 3. 清空队列
        queues_to_clear = [
            current_app.config.get('REQUEST_QUEUE'),
            current_app.config.get('TTS_QUEUE'),
            current_app.config.get('TTS_OUTPUT_QUEUE'),
            current_app.config.get('ACTION_FEATURE_QUEUE'),
            current_app.config.get('OBSERVATION_QUEUE'),
        ]
        print(f"{Colors.CYAN}清空队列...{Colors.RESET}")
        for queue in queues_to_clear:
            if queue:
                clear_queue(queue)
        
        # 4. 强制终止仍在运行的进程
        for name, process in processes_to_join.items():
            if process and process.is_alive():
                print(f"{Colors.RED}进程 {name} 未能正常退出，强制终止。{Colors.RESET}")
                process.terminate()
                process.join()

    print(f"{Colors.CYAN}资源清理完成。{Colors.RESET}")

if __name__ == "__main__":
    print(f"{Colors.CYAN}Start VITA server{Colors.RESET}")
    
    # 1. 初始化多进程相关资源
    multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    
    # 使用ProcessConfig类简化配置管理
    config = ProcessConfig(args, manager)

    # 2. 启动工作进程
    tts_worker_process = multiprocessing.Process(
        target=tts_worker,
        kwargs=config.get_tts_worker_kwargs()
    )

    model_1_process = multiprocessing.Process(
        target=load_model,
        kwargs=config.get_model_1_kwargs()
    )

    model_2_process = multiprocessing.Process(
        target=load_model,
        kwargs=config.get_model_2_kwargs()
    )

    # 3. 启动进程
    model_1_process.start()
    model_2_process.start()
    tts_worker_process.start()

    # 4. 将多进程资源添加到 Flask app context
    app_config = config.get_app_config()
    app_config.update({
        'MODEL_1_PROCESS': model_1_process,
        'MODEL_2_PROCESS': model_2_process,
        'TTS_WORKER_PROCESS': tts_worker_process,
    })
    
    for key, value in app_config.items():
        app.config[key] = value

    import cv2
    import pickle
    import torch
    import torchaudio
    
    # 5. 启动 Flask 应用
    cert_file = "vita_e/vita_vla_html/web/resources/cert.pem"
    key_file = "vita_e/vita_vla_html/web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    
    try:
        print(f"{Colors.GREEN}服务启动成功，请继续等待模型加载...{Colors.RESET}")
        socketio.run(app, host=args.ip, port=args.port, debug=False)  # , ssl_context=(cert_file, key_file)
    finally:
        print(f"{Colors.CYAN}捕获到退出信号，开始清理...{Colors.RESET}")
        cleanup_resources()

    # 6. 等待进程结束
    model_1_process.join()
    model_2_process.join()
    tts_worker_process.join()
