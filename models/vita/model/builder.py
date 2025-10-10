import os
import warnings


import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, logging


# 从项目的相对路径导入自定义的常量和模型类
from ..constants import GLOBAL_WEIGHTS_PATH
from ..model import *


# 将 Hugging Face Transformers 库的日志级别设置为 ERROR，以隐藏不必要的警告信息，使输出更简洁。
logging.set_verbosity_error()
# 忽略 Python 的所有警告信息。
warnings.filterwarnings("ignore")

def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    model_type,
    load_8bit=False,
    load_4bit=False,
    device_map=None,
    device="cuda",
    **kwargs,
):
    """
    加载预训练的多模态模型（VITA），支持多种加载方式和量化选项。

    该函数处理三种主要的加载场景：
    1. 从基础模型和 LoRA 权重加载并合并。
    2. 从基础模型加载，并附加预训练的多模态投影层 (mm_projector)。
    3. 加载一个已经完全合并好的、独立的多模态模型。
    """
    # --- 1. 参数校验和初始化 ---
    # 检查指定的模型类型是否在支持的列表中，如果不是则抛出错误。
    if model_type not in {"mixtral-8x7b", "nemo", "qwen2p5_instruct", "qwen2p5_fo_instruct"}:
        raise ValueError(f"Unknown Model Type {model_type}")


    # 将 device_map 和其他任意关键字参数合并，以便统一传递给 from_pretrained 方法。
    kwargs = {"device_map": device_map, **kwargs}


    # 如果目标设备不是 "cuda" (例如 "cpu" 或 "mps")，则强制将整个模型加载到该设备上。
    if device != "cuda":
        kwargs["device_map"] = {"": device}


    # --- 2. 配置量化参数 ---
    # 根据传入的标志位，配置不同的量化选项以节省显存。
    if load_8bit:
        # 如果为 True，则启用 8-bit 量化加载。
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        # 如果为 True，则启用 4-bit 量化加载。
        kwargs["load_in_4bit"] = True
        # 配置更详细的 BitsAndBytes 参数以优化 4-bit 量化。
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # 在计算过程中使用 float16 以保持精度。
            bnb_4bit_use_double_quant=True,       # 使用双重量化技术，进一步节省显存。
            bnb_4bit_quant_type="nf4",            # 使用 "nf4" (NormalFloat4) 量化类型，通常能提供更好的性能。
        )
    else:
        # 如果不进行量化，则默认使用 float16 数据类型，这是现代 GPU 推理的标准。
        kwargs["torch_dtype"] = torch.float16


    # --- 3. 根据不同场景加载模型 ---
    # 情况 A: 加载 LoRA 微调的模型。这需要一个基础模型 (model_base) 和 LoRA 适配器权重 (在 model_path 中)。
    if "lora" in model_name.lower() and model_base is None:
        # 如果模型名中包含 "lora" 但没有提供基础模型，则发出警告，因为这通常是错误的配置。
        warnings.warn(
            "There is `lora` in model name but no `model_base` is provided"
        )
    if "lora" in model_name.lower() and model_base is not None:
        # 从 LoRA 权重路径加载配置。
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        print("Loading VITA from base model...")
        # 根据模型类型，加载相应的基础 LLM 模型。
        if model_type == "mixtral-8x7b":
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            # 从 `model_base` 路径加载基础模型，并应用量化等配置。
            model = VITAMixtralForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            
        # 检查词嵌入层的大小，以防微调时添加了新的 token。如果大小不匹配，则重新初始化。
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
            )
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
            )


        # 加载在 LoRA 训练之外被更新的权重（例如，多模态投影层）。
        print("Loading additional VITA weights...")
        # 首先尝试从本地路径加载。
        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(
                os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu"
            )
        else:
            # 如果本地不存在，则认为 model_path 是 Hugging Face Hub 的仓库 ID，并尝试从中下载。
            from huggingface_hub import hf_hub_download


            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id, filename=filename, subfolder=subfolder
                )
                return torch.load(cache_file, map_location="cpu")


            non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")


        # 清理权重字典中的 key 前缀，以匹配当前模型的结构。
        non_lora_trainables = {
            (k[11:] if k.startswith("base_model.") else k): v
            for k, v in non_lora_trainables.items()
        }
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
            }
        # 加载这些非 LoRA 权重，strict=False 表示不要求所有 key 都完全匹配。
        model.load_state_dict(non_lora_trainables, strict=False)


        # 使用 PEFT 库来加载 LoRA 适配器权重。
        from peft import PeftModel


        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        # 将 LoRA 权重合并到基础模型中，并卸载适配器，得到一个标准的 Transformer 模型，这通常能提高推理速度。
        model = model.merge_and_unload()
        print("Model is loaded...")
    # # 情况 B: 仅加载多模态投影层。适用于有一个现成的 LLM，只想为其添加视觉/音频能力。
    # elif model_base is not None:
    #     print("Loading VITA from base model...")


    #     cfg_pretrained = AutoConfig.from_pretrained(model_path)
        # 加载基础 LLM 模型。
    #     if model_type == "mixtral-8x7b":
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    #         model = VITAMixtralForCausalLM.from_pretrained(
    #             model_base, low_cpu_mem_usage=True, **kwargs
    #         )


    #         # 初始化视觉模块（vision tower 和 projector）。
    #         from types import SimpleNamespace
    #         model_args = {
    #             "vision_tower": f"{GLOBAL_WEIGHTS_PATH}/InternViT-300M-448px",
    #             "pretrain_mm_mlp_adapter": None,
    #             "mm_projector_type": "mlp2x_gelu",
    #         }
    #         model_args = SimpleNamespace(**model_args)
    #         model.get_model().initialize_vision_modules(model_args=model_args)


    #         # 初始化音频模块。
    #         from types import SimpleNamespace
            # model_args = {
            #    'audio_encoder': f"{GLOBAL_WEIGHTS_PATH}/audio-encoder-2wh_
            #    zh_en_audioset_Mixtral-8x7B_New-base-tunning",
            #    'freeze_audio_encoder': True,
            #    'freeze_audio_encoder_adapter': True
            # }
    #         model_args = SimpleNamespace(**model_args)
    #         model.get_model().initialize_audio_modules(model_args=model_args)
    #         audio_encoder = model.get_audio_encoder()
    #         device = torch.device('cuda:0')
    #         audio_encoder = audio_encoder.to(device)


    #     # 从指定路径加载预训练的多模态投影层权重。
    #     mm_projector_weights = torch.load(
    #         os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
    #     )
    #     # 将权重转换为 float16 类型。
    #     mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    #     # 将权重加载到模型中。
    #     model.load_state_dict(mm_projector_weights, strict=False)
    #     # 确保新加载的模块在正确的设备和数据类型上。
    #     model.model.mm_projector.to(device="cuda", dtype=torch.float16)
    #     model.model.vision_tower.to(device="cuda", dtype=torch.float16)
    # # 情况 C: 加载一个完全合并好的独立模型。
    # else:
    #     # 根据不同的模型类型，执行相应的加载流程。
    #     if model_type == "mixtral-8x7b":
    #         # 为 Mixtral 模型手动指定一个设备映射，以在多 GPU 上平衡负载，防止 OOM。
    #         device_map = {
    #             "model.embed_tokens": 0,
    #             "model.layers.0": 0,
    #             "model.layers.1": 0,
    #             "model.layers.2": 0,
    #             "model.layers.3": 0,
    #             "model.layers.4": 0,
    #             "model.layers.5": 0,
    #             "model.layers.6": 0,
    #             "model.layers.7": 0,
    #             "model.layers.8": 0,
    #             "model.layers.9": 0,
    #             "model.layers.10": 0,
    #             "model.layers.11": 0,
    #             "model.layers.12": 0,
    #             "model.layers.13": 0,
    #             "model.layers.14": 0,
    #             "model.layers.15": 0,
    #             "model.layers.16": 1,
    #             "model.layers.17": 1,
    #             "model.layers.18": 1,
    #             "model.layers.19": 1,
    #             "model.layers.20": 1,
    #             "model.layers.21": 1,
    #             "model.layers.22": 1,
    #             "model.layers.23": 1,
    #             "model.layers.24": 1,
    #             "model.layers.25": 1,
    #             "model.layers.26": 1,
    #             "model.layers.27": 1,
    #             "model.layers.28": 1,
    #             "model.layers.29": 1,
    #             "model.layers.30": 1,
    #             "model.layers.31": 1,
    #             "model.norm": 1,
    #             "model.vision_tower": 1,
    #             "model.mm_projector": 1,
    #             "model.audio_encoder": 1,
    #             "lm_head": 1,
    #         }
    #         device_map["model.audio_encoder"] = 0
    #         kwargs.update(device_map=device_map)
    #         # 加载 tokenizer 和自定义的 VITA Mixtral 模型。
    #         tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #         model = VITAMixtralForCausalLM.from_pretrained(
    #             model_path, low_cpu_mem_usage=True, **kwargs
    #         )
    #     elif model_type == "nemo":
    #         # 加载 tokenizer 和自定义的 VITA Mistral 模型。
    #         tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #         model = VITAMistralForCausalLM.from_pretrained(
    #             model_path, low_cpu_mem_usage=True, **kwargs
    #         )
        # elif model_type == "qwen2p5_instruct":
        #     # 加载 tokenizer 和自定义的 VITA Qwen2 模型。
        #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        #     model = VITAQwen2ForCausalLM.from_pretrained(
        #         model_path, low_cpu_mem_usage=True, **kwargs
        #     )
        # elif model_type == "qwen2p5_fo_instruct":
        #     # 加载 tokenizer 和自定义的 VITA FO Qwen2 模型。
        #     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        #     model = VITAFOQwen2ForCausalLM.from_pretrained(
        #         model_path, low_cpu_mem_usage=True, **kwargs
        #     )

    elif model_type == "qwen2p5_instruct":
        # 加载 tokenizer 和自定义的 VITA Qwen2 模型。
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = VITAQwen2ForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet.")
    # --- 4. 后处理和最终化 ---
    # 确保模型的词嵌入层大小与 tokenizer 的词汇表大小一致。
    model.resize_token_embeddings(len(tokenizer))


    # 获取模型的视觉编码器（Vision Tower）。
    vision_tower = model.get_vision_tower()
    # 如果视觉编码器是懒加载的（is_loaded=False），则在此处实际加载其权重。
    if not vision_tower.is_loaded:
        vision_tower.load_model()


    # 计算并打印视觉编码器的参数量。
    num_params = sum(p.numel() for p in vision_tower.parameters())
    print("the number of vision encoder params: {}M".format(num_params / 1024 / 1024))


    # 如果配置要求 vision tower 是可训练的（即在微调中被更新），则加载其微调后的权重。
    if getattr(model.config, "unfreeze_vision_tower", False):
        if "lora" in model_name.lower():
            # 对于 LoRA 模型，从之前加载的 non_lora_trainables 字典中提取 vision tower 的权重。
            assert model_base is not None
            vision_non_lora_trainables = {
                k[19:]: v
                for k, v in non_lora_trainables.items()
                if k.startswith("model.vision_tower.")
            }
            vision_tower.load_state_dict(vision_non_lora_trainables, strict=False)
        else:
            # 对于完全合并的模型，从模型目录下的 safetensors 文件中加载 vision tower 的权重。
            assert model_base is None
            from safetensors.torch import load_file


            vision_weights = {}
            for file_name in os.listdir(model_path):
                if file_name.endswith("safetensors"):
                    vision_weights.update(
                        {
                            k[19:]: v
                            for k, v in load_file(os.path.join(model_path, file_name)).items()
                            if k.startswith("model.vision_tower.")
                        }
                    )
            vision_tower.load_state_dict(vision_weights, strict=True)


    # 确保 vision tower 的数据类型为 float16 以进行高效推理。
    vision_tower.to(dtype=torch.float16)
    # 获取与 vision tower 配套的图像处理器，用于预处理输入图像。
    image_processor = vision_tower.image_processor


    # 确定模型的上下文长度。
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        # 如果配置中没有，则使用一个默认值。
        context_len = 2048

    # 关键修复：如果 pad_token_id 未设置，在批量生成时可能会出错。
    # 通常将其设置为 eos_token_id 是一个安全的做法。
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # 针对特定模型的兼容性修复。
    if model_type == "phi-3":
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 返回所有必要的组件，以便进行后续的推理任务。
    return tokenizer, model, image_processor, context_len
