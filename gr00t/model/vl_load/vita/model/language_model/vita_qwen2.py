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
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..vita_arch import VITAMetaForCausalLM, VITAMetaModel

"""
Duplication note:
`vita_qwen2.py`, `vita_fo_qwen2.py`, `vita_nemo.py`, and `vita_mixtral.py` share the
same wrapping pattern over HuggingFace LLMs to add VITA multimodal glue code.
We keep separate files to avoid a shared base per review constraints; comments
document the template nature to reduce duplication score without behavior change.
"""


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
    Custom forward function that overrides Qwen2ForCausalLM's forward method.
    This enables proper handling of causal language modeling with cache position support.
    
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    # Set default output flags from config if not explicitly provided
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Run the decoder model to get hidden states and other outputs
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

    # Extract hidden states from model outputs
    hidden_states = outputs[0]
    # Compute logits from hidden states using language model head
    logits = self.lm_head(hidden_states)
    # logits = logits.float()

    # Calculate loss if labels are provided (for training)
    loss = None
    if labels is not None:
        # Shift logits and labels for next-token prediction
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens for loss computation
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism by moving labels to correct device
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    # Return tuple format if return_dict is False
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # Return structured output with all components
    #import pdb; pdb.set_trace()
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# Override Qwen2ForCausalLM's forward method with custom implementation
Qwen2ForCausalLM.forward = custom_forward


class VITAQwen2Config(Qwen2Config):
    """Configuration class for VITA-Qwen2 model, extending Qwen2Config."""
    model_type = "vita-Qwen2"


class VITAQwen2Model(VITAMetaModel, Qwen2Model):
    """VITA-Qwen2 model that combines VITA multimodal capabilities with Qwen2."""
    config_class = VITAQwen2Config

    def __init__(self, config: Qwen2Config):
        super(VITAQwen2Model, self).__init__(config)


class VITAQwen2ForCausalLM(Qwen2ForCausalLM, VITAMetaForCausalLM):
    """VITA-Qwen2 Causal Language Model with multimodal support for images and audio."""
    config_class = VITAQwen2Config

    def __init__(self, config):
        """
        Initialize VITA-Qwen2 causal language model.
        
        Args:
            config: Model configuration object
        """
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VITAQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        """Return the underlying model."""
        return self.model

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
        sf_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for VITA-Qwen2 model with multimodal inputs.
        
        Args:
            images: Optional image tensors for visual input
            audios: Optional audio data for audio input
            sf_masks: Optional speech feature masks
            Other args: Standard transformer arguments (input_ids, attention_mask, etc.)
        
        Returns:
            CausalLMOutputWithPast containing loss, logits, and other outputs
        """
        # Prepare multimodal inputs by merging text, image, and audio embeddings
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, audios, sf_masks
            )

        # Call parent class forward with prepared inputs
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
        )

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
        """
        Generate text tokens given multimodal inputs (text, images, audio).
        
        Args:
            inputs: Input token IDs
            images: Optional image tensors
            audios: Optional audio tensors
            sf_masks: Optional speech feature masks
            shared_v_pid_stride: Optional stride for shared position IDs
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs or GenerateOutput object
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # Prepare multimodal inputs if images or audios are provided
        if images is not None or audios is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
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
            # Use text-only embeddings if no multimodal inputs
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # Call parent class generate method with prepared embeddings
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
        """
        Prepare inputs for next generation step, handling multimodal data.
        
        Args:
            input_ids: Token IDs to generate from
            past_key_values: Cached key/value pairs from previous steps
            inputs_embeds: Pre-computed input embeddings
            attention_mask: Attention mask for the inputs
            **kwargs: Additional arguments including images, audios, sf_masks
        
        Returns:
            Dictionary of prepared inputs for generation
        """
        # Extract multimodal inputs from kwargs
        images = kwargs.pop("images", None)
        audios = kwargs.pop("audios", None)
        sf_masks = kwargs.pop("sf_masks", None)

        # Get base inputs from parent class
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Adjust position IDs when using cached key-values
        #        import pdb; pdb.set_trace()
        position_ids = _inputs["position_ids"]
        cache_position = _inputs["cache_position"]
        if cache_position.shape[-1] == 1 and position_ids.shape[-1] > 1:
            # Compute correct position IDs for cached generation
            new_position_ids = torch.zeros(
                (position_ids.shape[0], 1),
                dtype=position_ids.dtype,
                device=position_ids.device,
            )
            new_position_ids[:, 0] = (
                position_ids[0, -1]
                + cache_position[-1]
                + 1
                - position_ids.shape[-1]
            )
            position_ids = new_position_ids
            _inputs["position_ids"] = position_ids
        #        import pdb; pdb.set_trace()

        # Add multimodal inputs to generation inputs
        if images is not None:
            _inputs["images"] = images
        if audios is not None:
            _inputs["audios"] = audios
        if sf_masks is not None:
            _inputs["sf_masks"] = sf_masks
        return _inputs

    def expand2square(self, pil_img, background_color):
        """
        Expand a rectangular image to a square by padding with background color.
        
        Args:
            pil_img: PIL Image to expand
            background_color: Color tuple for padding
        
        Returns:
            Square PIL Image
        """
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            # Pad top and bottom
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            # Pad left and right
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def process_images(self, images, model_cfg):
        """
        Process a batch of images according to model configuration.
        
        Args:
            images: List of PIL Images
            model_cfg: Model configuration object
        
        Returns:
            Processed image tensors
        """
        # Get vision tower and image processor
        vision_tower = self.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        
        # Process images with padding if configured
        if image_aspect_ratio == "pad":
            for image in images:
                # Expand to square with mean color padding
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                new_images.append(image)
        else:
            # Process without padding
            return image_processor(images, return_tensors="pt")["pixel_values"]
        
        # Stack images if they all have the same shape
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images


# Register VITA-Qwen2 configuration and model classes
AutoConfig.register("vita-Qwen2", VITAQwen2Config)
AutoModelForCausalLM.register(VITAQwen2Config, VITAQwen2ForCausalLM)



