from typing import List, Optional, Tuple, Union


import torch
import torch.nn as nn
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput


# 导入项目内部定义的 VITA 架构基类
from ..vita_arch import VITAMetaForCausalLM, VITAMetaModel




def custom_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    这是一个自定义的前向传播函数，旨在替换 Hugging Face Transformers 库中
    `Qwen2ForCausalLM` 模型的原始 `forward` 方法。
    主要目的是在计算 logits 后不将其强制转换为 float32，以保持原始精度（如 float16 或 bfloat16），
    这对于混合精度训练和推理很重要。

    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            用于计算掩码语言模型损失的标签。索引应在 `[0, ..., config.vocab_size]` 范围内，
            或为 -100。索引为 `-100` 的 token 会被忽略（掩码），损失仅对标签在
            `[0, ..., config.vocab_size]` 范围内的 token 进行计算。

    Returns:
        Union[Tuple, CausalLMOutputWithPast]: 模型的输出，格式取决于 `return_dict` 参数。

    Example:
        ... (Hugging Face 风格的示例代码)
    """

    # 确定是否输出 attention 权重和隐藏状态，以及是否返回字典格式的输出
    # 如果用户未指定，则使用模型配置中的默认值。
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 调用底层的 transformer 模型 (self.model) 获取其输出（隐藏状态、past_key_values 等）
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    # 从模型输出中提取最终的隐藏状态
    hidden_states = outputs[0]
    # 使用语言模型头部 (lm_head) 将隐藏状态投影到词汇表空间，得到 logits
    logits = self.lm_head(hidden_states)
    # 原始的 Hugging Face 实现可能会在这里将 logits 转换为 float32。
    # 注释掉这一行是为了在混合精度训练/推理中保持 logits 的原始数据类型（如 float16），
    # 从而提高效率并减少内存使用。
    # logits = logits.float()

    # --- 计算损失 ---
    loss = None
    if labels is not None:
        # 为了预测下一个 token，将 logits 和 labels 错开一位
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 将 token 维度展平以便计算损失
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # 确保 labels 和 logits 在同一个设备上，这在分布式训练中很重要
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    # --- 格式化输出 ---
    # 如果不要求返回字典，则以元组的形式返回输出
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # 如果要求返回字典，则使用 Hugging Face 提供的标准输出类 CausalLMOutputWithPast
    # 这种方式更具可读性
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# --- Monkey Patching (猴子补丁) ---
# 这行代码在运行时动态地将 `Qwen2ForCausalLM` 类的 `forward` 方法
# 替换为我们上面定义的 `custom_forward` 函数。
# 这样做之后，程序中所有 `Qwen2ForCausalLM` 的实例都将使用我们修改后的逻辑。
Qwen2ForCausalLM.forward = custom_forward


# --- 自定义模型和配置的定义 ---

# 为 VITA-Qwen2 模型定义一个自定义的配置类
class VITAQwen2Config(Qwen2Config):
    # 设置一个独特的 `model_type` 字符串。
    # 这是为了将我们的自定义模型注册到 Hugging Face 的 AutoClass 系统中，
    # 使得 `AutoModel.from_pretrained` 等函数能够识别并正确加载我们的模型。
    model_type = "vita-Qwen2"


# 定义 VITA-Qwen2 的基础模型类
class VITAQwen2Model(VITAMetaModel, Qwen2Model):
    # 这个类同时继承了项目自定义的 `VITAMetaModel` 和 Hugging Face 的 `Qwen2Model`。
    # 这种多重继承的方式使得模型既能拥有 VITA 架构的特定功能，又能复用 Qwen2 的核心实现。
    config_class = VITAQwen2Config

    def __init__(self, config: Qwen2Config):
        # 调用父类（这里是 Qwen2Model）的构造函数来完成模型的初始化
        super(VITAQwen2Model, self).__init__(config)


class VITAQwen2ForCausalLM(Qwen2ForCausalLM, VITAMetaForCausalLM):
    config_class = VITAQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VITAQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.action_indices = None
        self.image_token_num = 49

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    # def get_action_indices(self):
    #     return self.action_indices

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audios: Optional[dict] = None,
        states: Optional[torch.FloatTensor] = None,
        action_pred_token: Optional[torch.Tensor] = None,
        action_pred_steps: Optional[int] = None,
        sf_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # image_primary=input_image_primary,
        # image_wrist=input_image_wrist,
        #         state=input_state,
        #         text_token=input_text_token,
        #         action=actions[:, :args.sequence_length, :]

        if inputs_embeds is None and action_pred_token is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_vlm(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, sf_masks
            )
        
        else:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                action_indices
            ) = self.prepare_inputs_labels_for_vla(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, 
                audios, states, action_pred_token, action_pred_steps, sf_masks
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        ), action_indices

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        sf_masks: Optional[torch.Tensor] = None,
        shared_v_pid_stride: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or audios is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_vlm(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                audios,
                sf_masks,
                shared_v_pid_stride,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        audios = kwargs.pop("audios", None)
        sf_masks = kwargs.pop("sf_masks", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

#        import pdb; pdb.set_trace()
        position_ids = _inputs["position_ids"]
        cache_position = _inputs["cache_position"]
        if cache_position.shape[-1] == 1 and position_ids.shape[-1] > 1:
            new_position_ids = torch.zeros((position_ids.shape[0],1), dtype=position_ids.dtype, 
                                           device=position_ids.device)
            new_position_ids[:, 0] = position_ids[0,-1] + cache_position[-1] + 1 - position_ids.shape[-1]
            position_ids = new_position_ids
            _inputs["position_ids"] = position_ids
#        import pdb; pdb.set_trace()

        if images is not None:
            _inputs["images"] = images
        if audios is not None:
            _inputs["audios"] = audios
        if sf_masks is not None:
            _inputs["sf_masks"] = sf_masks
        return _inputs

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, model_cfg):
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images


AutoConfig.register("vita-Qwen2", VITAQwen2Config)
AutoModelForCausalLM.register(VITAQwen2Config, VITAQwen2ForCausalLM)



