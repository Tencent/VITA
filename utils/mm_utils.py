import base64
import re
from io import BytesIO

import torch
from PIL import Image
from transformers import StoppingCriteria

# 导入预定义的特殊 token 索引
from .constants import AUDIO_TOKEN_INDEX, IMAGE_TOKEN_INDEX, ACTION_TOKEN_INDEX, STATE_TOKEN_INDEX


def load_image_from_base64(image):
    """
    将 base64 编码的字符串解码并加载为 PIL.Image 对象。

    Args:
        image (str): base64 编码的图像字符串。

    Returns:
        PIL.Image.Image: 解码后的图像对象。
    """
    # 将 base64 字符串解码为字节，然后使用 BytesIO 创建一个内存中的二进制文件流，最后用 PIL 打开
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """
    将一个 PIL 图像填充为正方形。

    通过在较短的一边添加指定背景色的边框，将图像居中放置，使其成为正方形。

    Args:
        pil_img (PIL.Image.Image): 输入的 PIL 图像。
        background_color (tuple): 用于填充的背景颜色，格式为 (R, G, B)。

    Returns:
        PIL.Image.Image: 填充为正方形后的图像。
    """
    width, height = pil_img.size
    if width == height:
        # 如果已经是正方形，直接返回
        return pil_img
    elif width > height:
        # 如果宽度大于高度，创建一个以宽度为边长的正方形画布
        result = Image.new(pil_img.mode, (width, width), background_color)
        # 将原图粘贴到画布中央（垂直居中）
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        # 如果高度大于宽度，创建一个以高度为边长的正方形画布
        result = Image.new(pil_img.mode, (height, height), background_color)
        # 将原图粘贴到画布中央（水平居中）
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    """
    使用指定的图像处理器处理一批图像。

    根据模型配置中的 `image_aspect_ratio` 决定处理策略。
    - 'pad': 将每张图片填充为正方形，然后进行预处理。
    - 其他: 直接使用 image_processor 对整批图像进行处理。

    Args:
        images (list[PIL.Image.Image]): 待处理的 PIL 图像列表。
        image_processor: Hugging Face 的图像处理器。
        model_cfg: 模型配置对象。

    Returns:
        torch.Tensor: 处理后并堆叠成一个批次的图像张量。
    """
    # 从模型配置中获取图像宽高比处理策略
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        # 如果策略是 'pad'，则逐个处理图像
        for image in images:
            # 将图像填充为正方形，背景色使用处理器的均值
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            # 使用图像处理器进行预处理（如缩放、归一化），并返回 PyTorch 张量
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        # 如果没有指定 'pad' 策略，直接使用处理器处理整个列表
        return image_processor(images, return_tensors="pt")["pixel_values"]
    
    # 如果所有处理后的图像形状都相同，将它们堆叠成一个批次张量
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    """
    将包含特殊 <image> token 的文本提示转换为 token ID 序列。

    它会将文本按 <image> 分割，分别进行 tokenize，然后在它们之间插入 image_token_index。

    Args:
        prompt (str): 包含 <image> 占位符的输入字符串。
        tokenizer: Hugging Face 的 tokenizer。
        image_token_index (int): 代表图像的特殊 token ID。
        return_tensors (str, optional): 如果为 'pt'，则返回 PyTorch 张量。

    Returns:
        list[int] or torch.Tensor: token ID 序列。
    """
    # 按 "<image>" 分割提示，并对每个文本块进行 tokenize
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        # 辅助函数：在列表 X 的元素之间插入分隔符 sep
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # 检查第一个文本块是否以 BOS (Beginning of Sentence) token 开头
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1  # 如果是，则设置偏移量为1，以跳过后续块中的 BOS token
        input_ids.append(prompt_chunks[0][0]) # 首先添加 BOS token

    # 在 tokenized 文本块之间插入图像 token ID
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        # 将处理后的块（跳过 BOS token）追加到最终的 input_ids 列表中
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            # 如果要求，返回 PyTorch 张量
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def tokenizer_image_audio_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    audio_token_index=AUDIO_TOKEN_INDEX,
    return_tensors=None,
):
    """
    将包含 <image> 和 <audio> 特殊 token 的文本提示转换为 token ID 序列。
    """
    prompt_chunks = []
    # 使用正则表达式按 <audio> 或 <image> 分割字符串，并保留分隔符
    for chunk in re.split(r"(<audio>|<image>)", prompt):
        if chunk == "<audio>":
            prompt_chunks.append([audio_token_index])
        elif chunk == "<image>":
            prompt_chunks.append([image_token_index])
        elif chunk: # 确保非空字符串
            prompt_chunks.append(tokenizer(chunk).input_ids)

    input_ids = []
    offset = 0
    # 同样处理 BOS token
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 遍历所有块，将它们拼接成最终的 token ID 序列
    for x in prompt_chunks:
        if x != [image_token_index] and x != [audio_token_index]:
            # 如果是文本块，跳过 BOS token
            input_ids.extend(x[offset:])
        else:
            # 如果是特殊 token，直接添加
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def tokenizer_image_action_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    action_token_index=ACTION_TOKEN_INDEX,
    state_token_index=STATE_TOKEN_INDEX,
    return_tensors=None,
):
    """
    将包含 <image>, <action>, <state> 特殊 token 的文本提示转换为 token ID 序列。
    逻辑与 `tokenizer_image_audio_token` 类似，只是支持更多的特殊 token。
    """
    prompt_chunks = []
    # 按 <action>, <image>, 或 <state> 分割字符串
    for chunk in re.split(r"(<action>|<image>|<state>)", prompt):
        if chunk == "<action>":
            prompt_chunks.append([action_token_index])
        elif chunk == "<image>":
            prompt_chunks.append([image_token_index])
        elif chunk == "<state>":
            prompt_chunks.append([state_token_index])
        elif chunk:
            prompt_chunks.append(tokenizer(chunk).input_ids)

    input_ids = []
    offset = 0
    # 处理 BOS token
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 拼接所有块
    for x in prompt_chunks:
        if x != [image_token_index] and x != [action_token_index] and x != [state_token_index]:
            input_ids.extend(x[offset:])
        else:
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    """
    从模型文件路径中提取一个简洁的模型名称。
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    # 如果路径以 'checkpoint-...' 结尾，则将倒数第二级目录名和最后一级目录名拼接
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        # 否则直接返回最后一级目录名
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    """
    一个自定义的停止准则，当生成的文本中出现指定的关键字时停止生成。
    """
    def __init__(self, keywords, tokenizer, input_ids):
        """
        初始化停止准则。

        Args:
            keywords (list[str]): 触发停止的关键字列表。
            tokenizer: Hugging Face 的 tokenizer。
            input_ids (torch.Tensor): 初始的输入 token ID。
        """
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        # 将所有关键字转换为 token ID
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            # 移除可能存在的 BOS token
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            # 记录最长关键字的长度，用于优化检查
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        # 记录初始输入的长度，只检查新生成的部分
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        对单个序列进行检查。
        """
        # 确定要检查的最新 token 的数量
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        
        # 检查1: 精确匹配 token ID 序列
        # 检查生成的序列末尾是否与任何一个关键字的 token ID 序列完全匹配
        for keyword_id in self.keyword_ids:
            # 截取输出序列的尾部，长度与当前关键字相同
            truncated_output_ids = output_ids[0, -keyword_id.shape[0] :]
            if torch.equal(truncated_output_ids, keyword_id):
                return True # 如果匹配，则停止
        
        # 检查2: 字符串匹配
        # 将最新生成的 token 解码回字符串，检查关键字是否作为子字符串出现
        # 这可以捕获一些由于 tokenization 差异而无法通过检查1的情况
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True # 如果找到，则停止
        
        return False # 如果没有找到任何关键字，则不停止

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        主调用函数，由 transformers 的 generate 方法调用。
        """
        outputs = []
        # 对批次中的每个序列分别调用检查函数
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        # 只有当批次中的所有序列都满足停止条件时，才返回 True
        return all(outputs)
