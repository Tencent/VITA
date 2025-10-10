import dataclasses
from enum import Enum, auto
from typing import List


 
# 定义一个枚举类，用于表示不同的大语言模型所使用的对话分隔符样式。
class SeparatorStyle(Enum):
    """Different separator style."""

    # 各种预定义的样式
    TWO = auto()
    PLAIN = auto()
    Nemo = auto()
    Qwen2p5Instruct = auto()
    MixtralZh = auto()
    MixtralTwo = auto()




@dataclasses.dataclass
class Conversation:
    """
    一个数据类，用于存储和管理整个对话历史。
    使用 dataclass 可以自动生成 __init__, __repr__ 等方法。
    """

    # --- 类的属性定义 ---
    
    # 系统提示（System Prompt），用于设定 AI 的角色和行为准则。
    # 可以是单个字符串，也可以是根据不同模态（如图像、视频、文本）选择的字符串列表。
    system: str
    # 对话中不同角色的名称列表，例如 ["USER", "ASSISTANT"]。
    roles: List[str]
    # 存储对话消息的列表。每个元素是一个列表，包含 [角色, 消息内容]。
    # 消息内容可以是字符串，也可以是包含 (文本, 图像, 图像处理模式) 的元组。
    messages: List[List[str]]
    # 对话历史的偏移量，通常用于在显示或处理时跳过某些初始消息。
    offset: int
    # 当前对话使用的分隔符样式，是上面定义的 SeparatorStyle 枚举的成员。
    sep_style: SeparatorStyle
    # 主要的分隔符，通常用于分隔不同轮次的对话。
    sep: str = "###"
    # 次要的分隔符，某些模型可能需要两个不同的分隔符。
    sep2: str = None
    # 模板的版本或名称，用于标识。
    version: str = "Unknown"

    # 一个标志位，用于控制是否跳过下一轮对话的处理（在此代码段中未直接使用，但可能用于更复杂的逻辑）。
    skip_next: bool = False


    def get_prompt(self, modality=None):
        """
        根据存储的对话历史和分隔符样式，生成最终输入给大语言模型的提示（Prompt）字符串。

        Args:
            modality (str, optional): 输入的模态，如 'image', 'video', 'lang'。
                                      用于选择合适的系统提示。Defaults to None.

        Returns:
            str: 格式化后的完整提示字符串。
        """
        messages = self.messages
        # --- 特殊处理多模态输入 ---
        # 如果第一条消息包含元组（表示有图像），则需要特殊处理。
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()  # 创建副本以避免修改原始消息列表
            init_role, init_msg = messages[0].copy()
            # 从消息文本中移除 <image> 占位符
            init_msg = init_msg[0].replace("<image>", "").strip()
            
            # 'mmtag' 版本是一种特殊的多模态格式
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                # 在对话开头插入特殊的图像标签和一条确认接收的消息
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                # 其他版本则将 <image> 占位符放在消息文本的开头
                messages[0] = (init_role, "<image>\n" + init_msg)

        # --- 根据不同的分隔符样式格式化 Prompt ---
        
        if self.sep_style == SeparatorStyle.Qwen2p5Instruct:
            # Qwen 模型的格式化逻辑，使用 <|im_start|> 和 <|im_end|> 作为标记
            wrap_qa = lambda msg: f"<|im_start|>{msg}<|im_end|>\n"
            wrap_qa2 = lambda msg: f"<|im_start|>{msg}<|im_end|>"
            seps = [self.sep, self.sep2]
            
            # 检查对话中是否包含图像
            has_image = False
            for i, (role, message) in enumerate(messages):
                if message and "<image>" in message:
                    has_image = True
                    break
            
            # 根据是否存在图像以及具体的模态，选择不同的系统提示
            # 注意：这里的注释 "临时使用，仍然需要更改" 表明这部分逻辑可能还在开发中
            if has_image:
                assert modality == "image" or modality == "video"
                if modality == "image" :
                    self.system = self.system[0] # 选择图像相关的系统提示
                elif modality == "video":
                    self.system = self.system[1] # 选择视频相关的系统提示
                else:
                    raise ValueError
            else:
                assert modality == "lang"
                self.system = self.system[2] # 选择纯语言的系统提示
            
            # 拼接最终的 Prompt
            ret = wrap_qa("system\n" + self.system)
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message # 如果是元组，只取文本部分
                    if i < len(messages) - 1:
                        ret += wrap_qa(role + '\n' + message)
                    else: # 最后一条消息使用不同的包装函数
                        ret += wrap_qa2(role + '\n' + message)
                else: # 如果消息为空，只添加角色和起始标记
                    ret += "<|im_start|>" + role + '\n'

        elif self.sep_style == SeparatorStyle.PLAIN:
            # 一种简单的格式，直接将系统提示和消息用分隔符拼接起来
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2] # 交替使用 sep 和 sep2
                else:
                    ret += ""
        else:
            # 如果遇到不支持的样式，则抛出异常
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret


    def append_message(self, role, message):
        """向对话历史中添加一条新消息。"""
        self.messages.append([role, message])


    def get_images(self, return_pil=False):
        """
        从对话历史中提取所有图像，并进行预处理。

        Args:
            return_pil (bool): 如果为 True，返回 PIL.Image 对象列表；
                               否则，返回 base64 编码的字符串列表。

        Returns:
            list: 包含处理后图像的列表。
        """
        images = []
        # 遍历从偏移量开始的消息
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            # 通常假设用户的消息（偶数索引）可能包含图像
            if i % 2 == 0:
                if type(msg) is tuple:
                    # 动态导入必要的库，避免在不需要时产生依赖
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    # 解包消息元组
                    msg, image, image_process_mode = msg
                    
                    # 根据指定的处理模式对图像进行操作
                    if image_process_mode == "Pad":
                        # 定义一个辅助函数，将图像填充为正方形
                        def expand2square(pil_img, background_color=(122, 116, 104)):
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
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        # 默认或裁剪模式，不做任何操作
                        pass
                    elif image_process_mode == "Resize":
                        # 调整图像大小到固定尺寸
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    # 根据参数决定返回 PIL 对象还是 base64 字符串
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images


    def to_gradio_chatbot(self):
        """
        将对话历史转换为 Gradio 的 Chatbot 组件所需的格式。
        格式为: [[user_msg, assistant_msg], [user_msg, assistant_msg], ...]
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0: # 用户消息
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    # --- 为在 Gradio UI 中显示而调整图像大小 ---
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    
                    # --- 将图像编码为 base64 并嵌入 HTML <img> 标签 ---
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = (
                        f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    )
                    # 将图像 HTML 和文本消息合并
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None]) # 添加用户消息，AI 回复暂时为 None
                else:
                    ret.append([msg, None]) # 如果是纯文本消息
            else: # AI 助手消息
                ret[-1][-1] = msg # 将 AI 的回复填充到上一条用户消息的对应位置
        return ret


    def copy(self):
        """创建并返回当前 Conversation 对象的一个深拷贝。"""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages], # 确保 messages 列表是新的
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


    def dict(self):
        """将 Conversation 对象序列化为字典。"""
        # 如果对话中包含图像（元组形式），则只保留文本部分
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        # 如果不包含图像，直接返回
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


# --- 以下是为不同模型预定义的对话模板 ---

conv_mixtral_zh = Conversation(
    system='''你是一个人工智能机器人。\n- 你是研究社区开发的大语言模型。
    你的设计宗旨是有益、诚实且无害。\n- 你支持使用用户选择的多种语言流利地进行交流并解答用户的问题。\n- 
    如果用户更正你生成的错误答案，你会向用户致歉并与用户探讨正确的答案。''',
    roles=("user", "bot"),
    version="mixtral_zh",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MixtralZh,
    sep="</s>",
    sep2="</s>",
)

conv_mixtral_two = Conversation(
    system=[
        '''You are an AI robot and your name is VITA. \n- 
        You are a multimodal large language model developed 
        by the open source community. Your aim is to be helpful, honest and harmless. \n
        - You support the ability to communicate fluently and answer user questions 
        in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user. \n- 
        You must answer the question strictly according to the content of the image given by the user, 
        and it is strictly forbidden to answer the question without the content of the image. 
        Please note that you are seeing the image, not the video.''',
        '''You are an AI robot and your name is VITA. 
        \n- You are a multimodal large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and answer user questions 
        in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user. \n- 
        You must answer the question strictly according to the content of the video given 
        by the user, and it is strictly forbidden to answer the question without 
        the content of the video. Please note that you are seeing the video, not the image.''',
        '''You are an AI robot and your name is VITA. \n- 
        You are a multimodal large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and answer user questions 
        in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user.''',
    ],
    roles=("user", "bot"),
    version="mixtral_two",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MixtralTwo,
    sep="</s>",
    sep2="</s>",
)

conv_nemo = Conversation(
    system=[
        '''You are an AI robot and your name is VITA. \n- 
        You are a multimodal large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and answer user questions 
        in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user. \n- 
        You must answer the question strictly according to the content of the image given 
        by the user, and it is strictly forbidden to 
        answer the question without the content of the image. Please note that 
        you are seeing the image, not the video.''',
        '''You are an AI robot and your name is VITA. \n- You are a multimodal 
        large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and answer user questions 
        in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user. \n- 
        You must answer the question strictly according to the 
        content of the video given by the user, and it is strictly forbidden to 
        answer the question without the content of the video. 
        Please note that you are seeing the video, not the image.''',
        '''You are an AI robot and your name is VITA. \n- 
        You are a multimodal large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and 
        answer user questions in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user.''',
    ],
    roles=("USER", "ASSISTANT"),
    version="nemo",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.Nemo,
    sep="[/INST]",
    sep2="</s>",
)

conv_qwen2p5_instruct = Conversation(
    system=[
        '''You are an AI robot and your name is VITA. \n- 
        Your aim is to be helpful, honest and harmless. \n- 
        You are a large language model in the field of embodied intelligence. 
        Please combine the macro action instructions(maybe None) with 
        the current state of the robotic arm (the first six values represent the arm's state, 
        and the seventh value represents the gripper's state), 
        as well as the observations from the camera (primary image, wrist image), 
        to provide me with the next action output. You need to do this step by step, 
        predicting only three future actions at each step. \n''',
        '''You are an AI robot and your name is VITA. \n- Your aim is to be helpful, 
        honest and harmless. \n- You are a large language model in the field of embodied intelligence. 
        Please combine the macro action instructions with the 
        current state of the robotic arm (the first six values represent the arm's state, 
        and the seventh value represents the gripper's state), 
        as well as the observations from the camera (primary image, wrist image), 
        to provide me with the next action output. You need to do this step by step, 
        predicting only three future actions at each step. \n- 
        You must answer the question strictly according to the content of the image given by the user, 
        and it is strictly forbidden to answer the question without the content of the image. 
        Please note that you are seeing the image, not the video.''',
        '''You are an AI robot and your name is VITA. \n- You are a multimodal 
        large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and answer user questions 
        in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user. \n- 
        You must answer the question strictly according to the content 
        of the video given by the user, and it is strictly forbidden to 
        answer the question without the content of the video. 
        Please note that you are seeing the video, not the image.''',
        '''You are an AI robot and your name is VITA. \n- 
        You are a multimodal large language model developed by the open source community. 
        Your aim is to be helpful, honest and harmless. \n- 
        You support the ability to communicate fluently and answer 
        user questions in multiple languages of the user's choice. \n- 
        If the user corrects the wrong answer you generated, 
        you will apologize and discuss the correct answer with the user.''',
    ],
    roles=("user", "assistant"),
    version="qwen2p5_instruct",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.Qwen2p5Instruct,
    sep="<|im_start|>",
    sep2="<|im_start|>",
)

conv_phi3 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_minicpm = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="minicpm",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="llama",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|end_of_text|>",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

default_conversation = conv_mixtral_two
conv_templates = {
    "default": conv_mixtral_two,
    "nemo": conv_nemo,
    "qwen2p5_instruct": conv_qwen2p5_instruct,
    "mixtral_zh": conv_mixtral_zh,
    "mixtral_two": conv_mixtral_two,
    "phi3": conv_phi3,
    "plain": conv_plain,
    "minicpm": conv_minicpm,
    "llama": conv_llama,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())

