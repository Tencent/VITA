import os
import re
import cv2
import pickle
import numpy as np
from PIL import Image
import torch
import torchaudio

# 定义颜色代码
class Colors:
    RED = '\033[91m'  # 快速识别错误：所有红色（RED）日志都代表需要关注的错误、中断或超时。
    GREEN = '\033[92m'  # 监控模型性能：绿色（GREEN）的日志可以帮助您了解 LLM 的处理时间和输出。
    YELLOW = '\033[93m'  # 追踪动作生成：黄色（YELLOW）的日志清晰地标示了机器人动作生成的完整流程。
    BLUE = '\033[94m'  # 观察数据流动：蓝色（BLUE）的日志让您能看到数据在不同队列和处理器之间的流动情况。
    MAGENTA = '\033[95m'  # 品红色（MAGENTA）用于那些临时的、详细的调试输出。
    CYAN = '\033[96m'  # 理解服务状态：青色（CYAN）的日志告诉您服务的启动、关闭和连接状态。
    WHITE = '\033[97m'
    RESET = '\033[0m'

# 全局配置常量
IMAGE_TOKEN_INDEX = 51000
AUDIO_TOKEN_INDEX = 51001
IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"
VIDEO_TOKEN = "<video>"

# 特殊token定义 - 直接使用符号
AUDIO_RESPONSE_TOKEN = "☞"  # 语音回复
TEXT_RESPONSE_TOKEN = "☜"   # 文字回复
ACT_TOKEN = "☝"             # 动作特征生成
HALT_TOKEN = "☀"           # 急停
INSTRUCTION_TOKEN = "☯"     # 指令分隔符
END_TOKEN = "☜"             # 动作结束
SPECIAL_TOKEN_LIST = [AUDIO_RESPONSE_TOKEN, TEXT_RESPONSE_TOKEN, ACT_TOKEN, HALT_TOKEN, END_TOKEN]

# TTS相关配置
decoder_topk = 2
codec_padding_size = 10
target_sample_rate = 16000

def load_model_embemding(model_path):
    """加载模型嵌入层"""
    from vita.model.language_model.vita_qwen2 import VITAQwen2Config, VITAQwen2ForCausalLM
    config_path = os.path.join(model_path, 'config.json')
    config = VITAQwen2Config.from_pretrained(config_path)
    model = VITAQwen2ForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True)
    embedding = model.get_input_embeddings()
    del model
    return embedding

def save_video(images, video_filename):
    """保存图像序列为视频文件"""
    if len(images) == 0:
        return
        
    copy_images = list(images)
    height, width, layers = copy_images[0].shape
    size = (width, height)
    print(f"Saving video with size {size}")

    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    for image in copy_images:
        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    out.release()

def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    """处理包含图像和音频token的提示词"""
    prompt_chunks = []
    for chunk in re.split(r'(<audio>|<image>)', prompt):
        if chunk == '<audio>':
            prompt_chunks.append([audio_token_index])
        elif chunk == '<image>':
            prompt_chunks.append([image_token_index])
        else:
            prompt_chunks.append(tokenizer(chunk).input_ids)
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in prompt_chunks:
        if x != [image_token_index] and x != [audio_token_index]:
            input_ids.extend(x[offset:])
        else:
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.LongTensor(input_ids)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def clear_queue(queue):
    """清空队列"""
    from queue import Empty
    while not queue.empty():
        try:
            queue.get_nowait()
        except Empty:
            break

def merge_current_and_history(
        global_history,
        current_request,
        skip_history_vision=False,
        skip_history_audio=False,
        move_image_token_to_start=False
    ):
    """合并当前请求和历史对话"""
    
    system_prompts = {
        "video": "<|im_start|>system\nYou are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.<|im_end|>\n",
        "image": "<|im_start|>system\nYou are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.<|im_end|>\n",
        "audio": "<|im_start|>system\nYou are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.<|im_end|>\n"
    }

    def select_system_prompt(current_request):
        if "multi_modal_data" in current_request:
            if "video" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "image" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "audio" in current_request["multi_modal_data"]:
                return system_prompts["audio"]
        return system_prompts["audio"]

    system_prompt = select_system_prompt(current_request)
    user_prefix = "<|im_start|>user\n"
    bot_prefix = "<|im_start|>assistant\n"
    eos = "<|im_end|>\n"

    if len(global_history) == 0:
        current_request["prompt"] = (system_prompt + user_prefix + current_request["prompt"] + eos + bot_prefix).replace('☞ ','☞').replace('☟ ','☟')
        return current_request
    
    # Initialize the current prompt and multimodal data
    current_prompt = system_prompt
    current_multi_modal_data = {"image": [], "audio": [], "video": []}

    # Add the history to the current prompt
    for history in global_history:
        assert "prompt" in history, "Prompt must be provided in history."
        assert "response" in history, "Response must be provided in history."

        history_prompt = history["prompt"]
        if skip_history_vision:
            history_prompt = history_prompt.replace(IMAGE_TOKEN, "").replace(VIDEO_TOKEN, "")
        if skip_history_audio:
            history_prompt = history_prompt.replace(AUDIO_TOKEN, "")
        
        history_prompt = user_prefix + history_prompt + eos + bot_prefix + history["response"] + eos
        for modality in ["image", "audio", "video"]:
            if (skip_history_vision and modality in ["image", "video"]) or (skip_history_audio and modality == "audio"):
                continue

            if "multi_modal_data" in history and modality in history["multi_modal_data"]:
                current_multi_modal_data[modality].extend(history["multi_modal_data"][modality])
        current_prompt += history_prompt
    
    # Add the current request to the current prompt
    current_prompt += user_prefix + current_request["prompt"] + eos + bot_prefix
    for modality in ["image", "audio", "video"]:
        if "multi_modal_data" in current_request and modality in current_request["multi_modal_data"]:
            current_multi_modal_data[modality].extend(current_request["multi_modal_data"][modality])
    
    for modality in ["image", "audio", "video"]:
        if current_multi_modal_data[modality] == []:
            current_multi_modal_data.pop(modality, None)
    
    if move_image_token_to_start:
        num_image_tokens = current_prompt.count(IMAGE_TOKEN)
        current_prompt = current_prompt.replace(IMAGE_TOKEN, "")
        current_prompt = current_prompt.replace(system_prompt, "")
        current_prompt = system_prompt + user_prefix + IMAGE_TOKEN * num_image_tokens + current_prompt.replace(user_prefix, '', 1)
    
    current_request["prompt"] = current_prompt.replace('☞ ','☞').replace('☟ ','☟')
    current_request["multi_modal_data"] = current_multi_modal_data

    return current_request

class ProcessConfig:
    """多进程配置管理类"""
    
    def __init__(self, args, manager):
        self.args = args
        self.manager = manager
        self._init_queues()
        self._init_events()
        self._init_shared_variables()
    
    def _init_queues(self):
        """初始化队列"""
        self.request_inputs_queue = self.manager.Queue()
        self.tts_inputs_queue = self.manager.Queue()
        self.tts_output_queue = self.manager.Queue()
        self.observation_queue = self.manager.Queue()
        self.action_feature_queue = self.manager.Queue()
        self.action_feature_stack = self.manager.list()
        self.observation_queue_lock = self.manager.Lock()
    
    def _init_events(self):
        """初始化事件"""
        self.shutdown_event = self.manager.Event()
        self.worker_1_stop_event = self.manager.Event()
        self.worker_2_stop_event = self.manager.Event()
        self.worker_1_start_event = self.manager.Event()
        self.worker_2_start_event = self.manager.Event()
        self.worker_1_start_event.set()
        self.worker_1_2_start_event_lock = self.manager.Lock()
        
        # 正在生成动作特征事件
        self.worker_1_is_generating_action_feature = self.manager.Event()
        self.worker_2_is_generating_action_feature = self.manager.Event()
        
        # 工作进程就绪事件
        self.llm_worker_1_ready = self.manager.Event()
        self.llm_worker_2_ready = self.manager.Event()
        self.tts_worker_ready = self.manager.Event()
        self.gradio_worker_ready = self.manager.Event()
    
    def _init_shared_variables(self):
        """初始化共享变量"""
        self.current_dialog = self.manager.dict({'value': ''})
        self.global_history = self.manager.list()
        self.global_history_limit = 1
    
    def get_model_1_kwargs(self):
        """获取模型1进程的参数"""
        return {
            "llm_id": 1,
            "model_path": self.args.model_path_vlm,
            "cuda_devices": "0",
            "inputs_queue": self.request_inputs_queue,
            "outputs_queue": self.tts_inputs_queue,
            "tts_outputs_queue": self.tts_output_queue,
            "action_feature_queue": self.action_feature_queue,
            "action_feature_stack": self.action_feature_stack,
            "start_event": self.worker_1_start_event,
            "other_start_event": self.worker_2_start_event,
            "start_event_lock": self.worker_1_2_start_event_lock,
            "stop_event": self.worker_1_stop_event,
            "other_stop_event": self.worker_2_stop_event,
            "is_generating_action_feature": self.worker_1_is_generating_action_feature,
            "other_is_generating_action_feature": self.worker_2_is_generating_action_feature,
            "current_dialog": self.current_dialog,
            "observation_queue": self.observation_queue,
            "observation_queue_lock": self.observation_queue_lock,
            "worker_ready": self.llm_worker_1_ready,
            "wait_workers_ready": [self.llm_worker_2_ready, self.tts_worker_ready],
            "global_history": self.global_history,
            "global_history_limit": self.global_history_limit,
            "gr00t_model_path": self.args.model_path_policy,
            "data_config_name": self.args.data_config_name,
            "shutdown_event": self.shutdown_event,
        }
    
    def get_model_2_kwargs(self):
        """获取模型2进程的参数"""
        return {
            "llm_id": 2,
            "model_path": self.args.model_path_vlm,
            "cuda_devices": "1",
            "inputs_queue": self.request_inputs_queue,
            "outputs_queue": self.tts_inputs_queue,
            "tts_outputs_queue": self.tts_output_queue,
            "action_feature_queue": self.action_feature_queue,
            "action_feature_stack": self.action_feature_stack,
            "start_event": self.worker_2_start_event,
            "other_start_event": self.worker_1_start_event,
            "start_event_lock": self.worker_1_2_start_event_lock,
            "stop_event": self.worker_2_stop_event,
            "other_stop_event": self.worker_1_stop_event,
            "is_generating_action_feature": self.worker_2_is_generating_action_feature,
            "other_is_generating_action_feature": self.worker_1_is_generating_action_feature,
            "current_dialog": self.current_dialog,
            "observation_queue": self.observation_queue,
            "observation_queue_lock": self.observation_queue_lock,
            "worker_ready": self.llm_worker_2_ready,
            "wait_workers_ready": [self.llm_worker_1_ready, self.tts_worker_ready],
            "global_history": self.global_history,
            "global_history_limit": self.global_history_limit,
            "gr00t_model_path": self.args.model_path_policy,
            "data_config_name": self.args.data_config_name,
            "shutdown_event": self.shutdown_event,
        }
    
    def get_tts_worker_kwargs(self):
        """获取TTS工作进程的参数"""
        return {
            "model_path": self.args.model_path_vlm,
            "inputs_queue": self.tts_inputs_queue,
            "outputs_queue": self.tts_output_queue,
            "worker_ready": self.tts_worker_ready,
            "wait_workers_ready": [self.llm_worker_1_ready, self.llm_worker_2_ready],
            "shutdown_event": self.shutdown_event,
        }
    
    def get_app_config(self):
        """获取Flask应用配置"""
        return {
            'REQUEST_QUEUE': self.request_inputs_queue,
            'TTS_QUEUE': self.tts_inputs_queue,
            'TTS_OUTPUT_QUEUE': self.tts_output_queue,
            'ACTION_FEATURE_QUEUE': self.action_feature_queue,
            'ACTION_FEATURE_STACK': self.action_feature_stack,
            'SHUTDOWN_EVENT': self.shutdown_event,
            'WORKER_1_STOP': self.worker_1_stop_event,
            'WORKER_2_STOP': self.worker_2_stop_event,
            'WORKER_1_START': self.worker_1_start_event,
            'WORKER_2_START': self.worker_2_start_event,
            'START_LOCK': self.worker_1_2_start_event_lock,
            'WORKER_1_READY': self.llm_worker_1_ready,
            'WORKER_2_READY': self.llm_worker_2_ready,
            'TTS_READY': self.tts_worker_ready,
            'GLOBAL_HISTORY': self.global_history,
            'CURRENT_DIALOG': self.current_dialog,
            'OBSERVATION_QUEUE': self.observation_queue,
            'OBSERVATION_QUEUE_LOCK': self.observation_queue_lock,
        } 