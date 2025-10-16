import argparse
import logging
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import six
import torch

from vl_load.vita.model.multimodal_encoder.whale.module.component.mamba import MambaSSM
from vl_load.vita.model.multimodal_encoder.whale.module.component.subsampling import Subsampling
from vl_load.vita.model.multimodal_encoder.whale.module.component.transformer import Transformer
from vl_load.vita.model.multimodal_encoder.whale.utils import make_pad_mask


def add_encoder_args(group):
    """
    Add encoder common arguments to argument parser.
    
    Args:
        group: Argument parser group to add arguments to
    
    Returns:
        Updated argument parser group
    """
    group.add_argument(
        "--encoder-layer-config",
        type=str,
        default="tdnn-dtc",
        help="Layer config of encoder. Format layername-layername-..., default(conv1d-fsmn-rnn)",
    )
    group.add_argument(
        "--encoder-input-dim",
        type=int,
        default=256,
        help="Input dim of encoder. Must equal to the input dim of the first Component (default=40)",
    )
    group.add_argument(
        "--encoder-output-dim",
        type=int,
        default=256,
        help="Output dim of encoder. Must enqual to the output dim of the last Component ! (default=256)",
    )
    # Add args of all kinds of components.
    # If you add a new component, DO NOT forget to add args to add_component_args func.
    group = Transformer.add_arguments(group)
    group = Subsampling.add_arguments(group)
    group = MambaSSM.add_arguments(group)
    return group


def assign_args_from_dict(args, dict, prefix_key=None):
    """
    Assign values from a dictionary to argument namespace.
    
    Args:
        args: Argument namespace to update
        dict: Dictionary containing values to assign
        prefix_key: Optional key to access nested dictionary
    
    Returns:
        Updated argument namespace
    """
    if prefix_key is not None:
        dict = dict[prefix_key]
    for k, v in dict.items():
        # Convert hyphenated keys to underscore format
        k_args = k.replace("-", "_")
        if hasattr(args, k_args):
            setattr(args, k_args, dict[k])
    return args


class whaleEncoder(torch.nn.Module):
    """
    Whale encoder module for audio processing.
    Supports transformer, subsampling, and Mamba SSM components in a configurable stack.
    """
    def __init__(self, input_dim, overview_conf=None, para_conf=None, global_cmvn=None):
        """
        Initialize the Whale encoder with configurable layers.
        
        Args:
            input_dim: Input feature dimension
            overview_conf: Overview configuration dictionary
            para_conf: Per-component parameter configuration dictionary
            global_cmvn: Global CMVN (Cepstral Mean and Variance Normalization) module
        """
        super(whaleEncoder, self).__init__()

        # Parse encoder arguments
        parser = argparse.ArgumentParser()
        add_encoder_args(parser)
        args, _ = parser.parse_known_args()

        # Assign overview configuration
        assign_args_from_dict(args, overview_conf)
        # assign_args_from_dict(args, para_conf)

        # Parse layer configuration and build encoder stack
        self.config = args.encoder_layer_config.split("-")
        encoder_input_dim = args.encoder_input_dim
        encoder_output_dim = args.encoder_output_dim
        prev_output_dim = encoder_input_dim
        prev_component_name = "encoder"
        self.enc = torch.nn.ModuleList([])
        
        # Build each layer based on configuration
        for name in self.config:
            assign_args_from_dict(args, para_conf[name])
            # Extract base component name (remove suffix if present)
            if len(name.split("_")) == 2:
                name = name.split("_")[0]
            elif len(name.split("_")) == 1:
                name = name
            else:
                logging.error("WRONG CONFIG! {} is not valid".format("encoder", name))
                sys.exit()

            # Instantiate the appropriate component
            if name == "transformer":
                self.enc.append(Transformer(args))
            elif name == "subsampling":
                self.enc.append(Subsampling(args))
            elif name == "mamba":
                self.enc.append(MambaSSM(args))
            else:
                print("{} is not supported now!".format(name))
                return NotImplemented
            
            # Validate dimension compatibility between consecutive layers
            component_input_dim = getattr(args, name + "_input_dim")
            if component_input_dim != prev_output_dim:
                # This is the first layer
                logging.error(
                    "WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-input-dim ({})".format(
                        prev_component_name, prev_output_dim, name, component_input_dim
                    )
                )
                sys.exit()
            prev_output_dim = getattr(args, name + "_output_dim")
            prev_component_name = name

        self.global_cmvn = global_cmvn
        
        # Validate final output dimension
        if prev_output_dim != encoder_output_dim:
            logging.error(
                "WRONG CONFIG! --{}-output-dim ({}) does not equal to --{}-output-dim ({}, the last component)".format(
                    "encoder", encoder_output_dim, name, prev_output_dim
                )
            )
            sys.exit()

        self._output_size = encoder_output_dim

        # Print parameter count
        num_params = sum(p.numel() for p in self.parameters())
        print("the number of whale encoder params: {}M".format(num_params / 1024 / 1024))

    def output_size(self) -> int:
        """Return the output dimension of the encoder."""
        return self._output_size

    @torch.jit.unused
    def forward(self, xs, ilens, decoding_chunk_size=None, num_decoding_left_chunks=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Optional[List[int]], Optional[Tensor]]
        """
        Encoder forward pass for audio feature processing.

        Args:
            xs: Batch of padded input sequences (B, Tmax, D)
            ilens: Batch of lengths of input sequences (B)
            decoding_chunk_size: Optional chunk size for streaming decoding
            num_decoding_left_chunks: Optional number of left chunks for context
        
        Returns:
            xs: Batch of hidden state sequences (B, Tmax, eprojs)
            masks: Attention masks for the sequences
        """

        # Configure chunk-based decoding parameters if specified
        if decoding_chunk_size is not None and num_decoding_left_chunks is not None:
            for layer in self.enc:
                if hasattr(layer, "chunk_size"):
                    layer.chunk_size = decoding_chunk_size
                if hasattr(layer, "left_chunks"):
                    layer.left_chunks = num_decoding_left_chunks
                if hasattr(layer, "transformer_dynamic_chunks"):
                    layer.transformer_dynamic_chunks = False

        # Validate input shape and create padding masks
        assert (len(xs.shape)) == 3
        T = xs.size(1)
        masks = ~make_pad_mask(ilens, T).unsqueeze(1)  # (B, 1, T)
        
        # Apply global CMVN normalization if available
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        
        # Process through encoder layers
        for module in self.enc:
            xs, ilens, masks = module(xs, ilens, masks)
        return xs, masks

    @torch.jit.export
    def infer(self, xs_pad, buffer, buffer_index, buffer_out):
        """
        Streaming inference with buffering for real-time processing.
        
        Args:
            xs_pad: Padded input sequences
            buffer: Buffer for caching intermediate states
            buffer_index: Current buffer position index
            buffer_out: Output buffer
        
        Returns:
            Tuple of (processed sequences, updated buffer, updated index, updated output buffer)
        """
        # Apply CMVN normalization if available
        if self.global_cmvn is not None:
            xs_pad = self.global_cmvn(xs_pad)
        
        # Process through encoder layers with buffering
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out = module.infer(
                xs_pad, buffer, buffer_index, buffer_out
            )
        return xs_pad, buffer, buffer_index, buffer_out

    @torch.jit.export
    def infer_hidden(self, xs_pad, buffer, buffer_index, buffer_out, hidden_out):
        """
        Streaming inference with hidden state output for analysis or debugging.
        
        Args:
            xs_pad: Padded input sequences
            buffer: Buffer for caching intermediate states
            buffer_index: Current buffer position index
            buffer_out: Output buffer
            hidden_out: Hidden state output buffer
        
        Returns:
            Tuple of (processed sequences, updated buffer, updated index, updated output buffer, hidden states)
        """
        # Apply CMVN normalization if available
        if self.global_cmvn is not None:
            xs_pad = self.global_cmvn(xs_pad)
        
        # Process through encoder layers with hidden state tracking
        for module in self.enc:
            xs_pad, buffer, buffer_index, buffer_out, hidden_out = module.infer_hidden(
                xs_pad, buffer, buffer_index, buffer_out, hidden_out
            )
        return xs_pad, buffer, buffer_index, buffer_out, hidden_out

    @torch.jit.ignore(drop=True)
    def get_extra_loss(self) -> Dict[str, torch.Tensor]:
        """
        Get extra losses from encoder components (e.g., regularization losses).
        
        Returns:
            Dictionary of extra losses (currently returns None)
        """
        return None
