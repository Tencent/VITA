"""Encoder self-attention layer definition."""


import math
import pdb
from functools import partial


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from ......model.multimodal_encoder.whale.utils import IGNORE_ID, strtobool


# MambaSSM is a third-party library. This block ensures the code can be imported
# even if mamba_ssm is not installed, printing a helpful message.
try:
    from mamba_ssm.modules.mamba_simple import Mamba, Block
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    print("Please install mamba_ssm to use MambaSSM component.")




class MambaBlock(nn.Module):
    """
    A block of Mamba layers, which can be configured to be unidirectional or bidirectional.
    This module stacks multiple Mamba layers and handles the forward and optional backward passes.
    """
    def __init__(self, in_channels, n_layer=1, d_state=16, d_conv=4, expand=4, bidirectional=False):
        """
        Initializes the MambaBlock.

        Args:
            in_channels (int): The number of input channels (embedding dimension).
            n_layer (int): The number of Mamba layers to stack.
            d_state (int): The dimension of the state space model's state (N).
            d_conv (int): The dimension of the 1D convolution kernel.
            expand (int): The expansion factor for the hidden dimension.
            bidirectional (bool): If True, process the sequence in both forward and backward directions.
        """
        super(MambaBlock, self).__init__()
        # ModuleList to hold the forward-pass Mamba blocks
        self.forward_blocks = nn.ModuleList([])
        # Final normalization layer for the forward pass
        self.forward_norm_f = RMSNorm(in_channels, eps=1e-5)
        # Create and append n_layer Mamba blocks for the forward pass
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    # Use partial to pre-configure the Mamba mixer class
                    mixer_cls=partial(
                        Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand
                    ),
                    # Use partial to pre-configure the RMSNorm normalization class
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=True,  # Use a fused kernel for efficiency
                    residual_in_fp32=True, # Perform residual addition in float32 for precision
                )
            )
        
        # If bidirectional, create a separate set of blocks for the backward pass
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mixer_cls=partial(
                            Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand
                        ),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=True,
                        residual_in_fp32=True,
                    )
                )
            # Final normalization layer for the backward pass
            self.backward_norm_f = RMSNorm(in_channels, eps=1e-5)
        else:
            self.backward_blocks = None

        # Apply the custom weight initialization from the mamba_ssm library
        self.apply(partial(_init_weights, n_layer=n_layer))


    def forward(self, input):
        """
        Forward pass through the MambaBlock.

        Args:
            input (torch.Tensor): The input tensor of shape (batch, seq_len, in_channels).

        Returns:
            torch.Tensor: The output tensor. If bidirectional, the output dimension will be doubled.
        """
        # --- Forward Pass ---
        for_residual = None
        forward_f = input.clone() # Create a copy for the forward pass
        # Pass the input through each Mamba block in the forward direction
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        # Add the final residual connection
        residual = (forward_f + for_residual) if for_residual is not None else forward_f
        # Apply the final normalization
        residual = self.forward_norm_f(residual)

        # --- Backward Pass (if enabled) ---
        if self.backward_blocks is not None:
            back_residual = None
            # Reverse the sequence along the time dimension (dim=1) for the backward pass
            backward_f = torch.flip(input, [1])
            # Pass the flipped input through each Mamba block in the backward direction
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            # Add the final residual connection for the backward pass
            back_residual = (
                (backward_f + back_residual) if back_residual is not None else backward_f
            )

            # Flip the result back to the original sequence order
            back_residual = torch.flip(back_residual, [1])
            # Apply the final normalization for the backward pass
            back_residual = self.backward_norm_f(back_residual)
            # Concatenate the forward and backward pass outputs along the feature dimension
            residual = torch.cat([residual, back_residual], -1)

        return residual




class MambaSSM(torch.nn.Module):
    """
    A wrapper module for the MambaBlock, designed to be easily integrated into a larger model
    and configured via command-line arguments.
    """
    @staticmethod
    def add_arguments(group):
        """
        Adds MambaSSM-specific command-line arguments to an argument parser group.
        This is a common pattern for creating modular and configurable models.
        """
        group.add_argument(
            "--mamba-num-layers", default=4, type=int, help="Number of Mamba layers."
        )
        group.add_argument(
            "--mamba-input-dim", default=256, type=int, help="Input dimension of MambaSSM."
        )
        group.add_argument(
            "--mamba-output-dim", default=256, type=int, help="Output dimension of MambaSSM."
        )
        group.add_argument("--mamba-d-state", default=16, type=int, help="State dimension (N) of MambaSSM.")
        group.add_argument("--mamba-d-conv", default=4, type=int, help="Convolution kernel size of MambaSSM.")
        group.add_argument("--mamba-expand", default=4, type=int, help="Expansion factor of MambaSSM.")
        return group


    def __init__(self, args):
        """
        Construct a MambaSSM object from parsed arguments.
        
        Args:
            args: An object (e.g., from argparse) containing the model hyperparameters.
        """
        super(MambaSSM, self).__init__()
        # Store hyperparameters from the args object
        self.mamb_num_layers = args.mamba_num_layers
        self.mamba_input_dim = args.mamba_input_dim
        self.mamba_output_dim = args.mamba_output_dim
        self.mamba_d_state = args.mamba_d_state
        self.mamba_d_conv = args.mamba_d_conv
        self.mamba_expand = args.mamba_expand

        # Instantiate the core MambaBlock with the specified configuration.
        # Note: This configuration is unidirectional as `bidirectional` is not passed.
        self.mamba = MambaBlock(
            self.mamba_input_dim,
            self.mamb_num_layers,
            self.mamba_d_state,
            self.mamba_d_conv,
            self.mamba_expand,
        )


    @torch.jit.unused
    def forward(self, xs, ilens=None, masks=None):
        """
        Defines the forward pass of the MambaSSM module.

        Args:
            xs (torch.Tensor): The input tensor (batch, seq_len, features).
            ilens (torch.Tensor, optional): A tensor of sequence lengths.
            Not used by Mamba but kept for API compatibility.
            masks (torch.Tensor, optional): An input mask. Not used by Mamba but kept for API compatibility.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the output tensor,
            and the original ilens and masks passed through.
        """

        # Pass the input tensor through the MambaBlock
        xs_out = self.mamba(xs)

        # Ensure output dtype matches input dtype and return along with passthrough variables
        return xs_out.to(xs.dtype), ilens, masks
