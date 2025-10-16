"""
Conversation module for VITA multimodal large language model.
This module handles conversation history, message formatting, and prompt generation.
"""

import dataclasses
from enum import Enum, auto
from typing import List


class SeparatorStyle(Enum):
    """Different separator style for conversation formatting."""

    TWO = auto()  # Two-part separator style
    PLAIN = auto()  # Plain text separator
    Nemo = auto()  # Nemo model separator style
    Qwen2p5Instruct = auto()  # Qwen2.5 Instruct model separator style
    MixtralZh = auto()  # Mixtral Chinese separator style
    MixtralTwo = auto()  # Mixtral two-part separator style


@dataclasses.dataclass
class Conversation:
    """
    A class that keeps all conversation history and manages message formatting.
    
    Attributes:
        system: System prompt that defines the AI assistant's behavior and capabilities
        roles: List of role names (e.g., ["user", "assistant"])
        messages: List of message pairs [role, content]
        offset: Starting index for message iteration
        sep_style: Style of separator used for formatting (from SeparatorStyle enum)
        sep: Primary separator string
        sep2: Secondary separator string (optional)
        version: Model version identifier
        skip_next: Flag to skip the next message in processing
    """

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self, modality=None):
        """
        Generate formatted prompt string from conversation history.
        
        Args:
            modality: Type of input modality ("image", "video", or "lang")
        
        Returns:
            Formatted prompt string ready for model input
        """
        messages = self.messages
        # Handle messages with embedded images/videos
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            # Handle multimodal tag version
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        # Format messages according to Qwen2.5 Instruct style
        if self.sep_style == SeparatorStyle.Qwen2p5Instruct:
            # Lambda functions to wrap messages with start/end tags
            wrap_qa = lambda msg: f"<|im_start|>{msg}<|im_end|>\n"
            wrap_qa2 = lambda msg: f"<|im_start|>{msg}<|im_end|>"
            seps = [self.sep, self.sep2]
            # Check if conversation contains image/video content
            has_image = False
            for i, (role, message) in enumerate(messages):
                if message and "<image>" in message:
                    has_image = True
                    break
            # Select appropriate system prompt based on modality
            if has_image:
                assert modality == "image" or modality == "video"
                if modality == "image":
                    self.system = self.system[0]
                elif modality == "video":
                    self.system = self.system[1]
                else:
                    raise ValueError
            else:
                # For text-only conversations
                assert modality == "lang"
                self.system = self.system[2]
            # Build the complete prompt
            ret = wrap_qa("system\n" + self.system)
            for i, (role, message) in enumerate(messages):
                if message:
                    # Extract message content if it's a tuple
                    if type(message) is tuple:
                        message, _, _ = message
                    # Add appropriate wrapper based on position
                    if i < len(messages) - 1:
                        ret += wrap_qa(role + '\n' + message)
                    else:
                        # Last message uses different wrapper (no trailing newline)
                        ret += wrap_qa2(role + '\n' + message)
                else:
                    # Empty message, add role only
                    ret += "<|im_start|>" + role + '\n'

        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        """
        Append a new message to the conversation history.
        
        Args:
            role: Role of the message sender (e.g., "user" or "assistant")
            message: Content of the message
        """
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        """
        Extract and process images from conversation messages.
        
        Args:
            return_pil: If True, return PIL Image objects; if False, return base64 encoded strings
        
        Returns:
            List of processed images (either PIL Images or base64 strings)
        """
        images = []
        # Iterate through messages starting from offset
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            # Process user messages only (even indices)
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    msg, image, image_process_mode = msg
                    # Apply different processing modes
                    if image_process_mode == "Pad":
                        # Expand image to square with padding
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
                        # No processing needed for default or crop mode
                        pass
                    elif image_process_mode == "Resize":
                        # Resize to fixed dimensions
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    # Return either PIL image or base64 encoded string
                    if return_pil:
                        images.append(image)
                    else:
                        # Convert to base64 string
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        """
        Convert conversation history to Gradio chatbot format.
        Images are resized and encoded as base64 strings for display.
        
        Returns:
            List of [user_message, assistant_message] pairs for Gradio chatbot UI
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            # Process user messages (even indices)
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    # Calculate aspect ratio for resizing
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    # Maintain aspect ratio while resizing
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # Convert to base64 for HTML display
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = (
                        f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    )
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    # Text-only message
                    ret.append([msg, None])
            else:
                # Assistant message (odd indices)
                ret[-1][-1] = msg
        return ret

    def copy(self):
        """
        Create a deep copy of the conversation object.
        
        Returns:
            A new Conversation instance with the same attributes
        """
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        """
        Convert conversation to dictionary format for serialization.
        Handles both text-only and multimodal messages.
        
        Returns:
            Dictionary representation of the conversation
        """
        # If images exist, extract only text content from tuple messages
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        # For text-only conversations
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


# Qwen2.5 Instruct conversation template for VITA
# Includes three system prompts for different modalities: image, video, and language-only
conv_qwen2p5_instruct = Conversation(
    system=[
        (
            "You are an AI robot and your name is VITA. \n"
            "- You are a multimodal large language model developed by the open source community. "
            "Your aim is to be helpful, honest and harmless. \n"
            "- You support the ability to communicate fluently and answer user questions "
            "in multiple languages of the user's choice. \n"
            "- If the user corrects the wrong answer you generated, you will apologize "
            "and discuss the correct answer with the user. \n"
            "- You must answer the question strictly according to the content of the image "
            "given by the user, and it is strictly forbidden to answer the question "
            "without the content of the image. Please note that you are seeing the image, "
            "not the video."
        ),
        (
            "You are an AI robot and your name is VITA. \n"
            "- You are a multimodal large language model developed by the open source community. "
            "Your aim is to be helpful, honest and harmless. \n"
            "- You support the ability to communicate fluently and answer user questions "
            "in multiple languages of the user's choice. \n"
            "- If the user corrects the wrong answer you generated, you will apologize "
            "and discuss the correct answer with the user. \n"
            "- You must answer the question strictly according to the content of the video "
            "given by the user, and it is strictly forbidden to answer the question "
            "without the content of the video. Please note that you are seeing the video, "
            "not the image."
        ),
        (
            "You are an AI robot and your name is VITA. \n"
            "- You are a multimodal large language model developed by the open source community. "
            "Your aim is to be helpful, honest and harmless. \n"
            "- You support the ability to communicate fluently and answer user questions "
            "in multiple languages of the user's choice. \n"
            "- If the user corrects the wrong answer you generated, you will apologize "
            "and discuss the correct answer with the user."
        ),
    ],
    roles=("user", "assistant"),
    version="qwen2p5_instruct",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.Qwen2p5Instruct,
    sep="<|im_start|>",
    sep2="<|im_start|>",
)

# Default conversation template used by VITA
default_conversation = conv_qwen2p5_instruct

# Dictionary of available conversation templates
conv_templates = {
    "qwen2p5_instruct": conv_qwen2p5_instruct,
}

if __name__ == "__main__":
    # Test: print the default conversation prompt
    print(default_conversation.get_prompt())

