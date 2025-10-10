import torch
import torch.nn as nn
import torch.nn.functional as F

class FsmnLayer(nn.Module):
    """
    Feedforward Sequential Memory Network (FSMN) Layer.

    This layer is designed to model long-term dependencies in sequential data
    by incorporating a learnable look-back and look-ahead context. It uses
    1D convolutions (specifically, depthwise convolutions) to efficiently
    compute a weighted sum of features from neighboring time steps.

    Args:
        input_dim (int): The number of expected features in the input.
        out_dim (int): The number of features in the output.
        hidden_dim (int): The internal dimension of the layer.
        left_frame (int): The number of context frames to look at from the past.
        right_frame (int): The number of context frames to look at from the future.
        left_dilation (int): The dilation factor for the left context convolution.
        right_dilation (int): The dilation factor for the right context convolution.
    """
    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_dim,
        left_frame=1,
        right_frame=1,
        left_dilation=1,
        right_dilation=1,
    ):
        super(FsmnLayer, self).__init__()
        # Store model dimensions and parameters
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.left_dilation = left_dilation
        self.right_dilation = right_dilation

        # A 1x1 convolution acts as a linear projection layer to transform the input
        # from input_dim to the internal hidden_dim.
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Setup for left (past) context aggregation
        if left_frame > 0:
            # Pad the sequence on the left to ensure the convolution can be applied
            # at the beginning of the sequence.
            self.pad_left = nn.ConstantPad1d([left_dilation * left_frame, 0], 0.0)
            # A depthwise 1D convolution to learn weights for past context frames.
            # `groups=hidden_dim` makes it a depthwise convolution, where each channel
            # is convolved with its own filter, efficiently learning temporal patterns
            # for each feature independently.
            self.conv_left = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=left_frame + 1,
                dilation=left_dilation,
                bias=False,
                groups=hidden_dim,
            )

        # Setup for right (future) context aggregation
        if right_frame > 0:
            # Pad the sequence on the right for the look-ahead convolution.
            # The negative padding on the left side is used to align the convolution window correctly.
            self.pad_right = nn.ConstantPad1d([-right_dilation, right_dilation * right_frame], 0.0)
            # A depthwise 1D convolution to learn weights for future context frames.
            self.conv_right = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=right_frame,
                dilation=right_dilation,
                bias=False,
                groups=hidden_dim,
            )

        # A final 1x1 convolution to project the aggregated features from hidden_dim
        # back to the desired output dimension.
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)

        # --- Cache and buffer sizes for streaming inference (`infer` method) ---
        # These are pre-calculated to manage the state buffer efficiently during
        # step-by-step inference.

        # Total number of context frames to cache for the main convolution input.
        self.cache_size = left_frame * left_dilation + right_frame * right_dilation
        # Total size of the buffer required for the main cache.
        self.buffer_size = self.hidden_dim * self.cache_size

        # Cache size for the raw projected input (p_in_raw), needed for future context.
        self.p_in_raw_chache_size = self.right_frame * self.right_dilation
        self.p_in_raw_buffer_size = self.hidden_dim * self.p_in_raw_chache_size

        # Cache size for the hidden state from a previous layer, also for future context.
        self.hidden_chache_size = self.right_frame * self.right_dilation
        self.hidden_buffer_size = self.hidden_dim * self.hidden_chache_size

    @torch.jit.unused
    def forward(self, x, hidden=None):
        """
        Standard forward pass for training or batch processing.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            hidden (Tensor, optional): Hidden state from a previous FSMN layer. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]:
                - The output tensor of shape (batch_size, seq_len, out_dim).
                - The intermediate tensor `p_out` before the final projection, which can be
                  used as the `hidden` input for the next layer.
        """
        # Conv1d expects (batch, channels, length), so we transpose the input.
        x_data = x.transpose(1, 2)
        
        # Project input to hidden dimension.
        p_in = self.conv_in(x_data)

        # Calculate left context (memory from the past).
        if self.left_frame > 0:
            p_left = self.pad_left(p_in)
            p_left = self.conv_left(p_left)
        else:
            p_left = 0

        # Calculate right context (look-ahead into the future).
        if self.right_frame > 0:
            p_right = self.pad_right(p_in)
            p_right = self.conv_right(p_right)
        else:
            p_right = 0

        # Combine current input projection with past and future contexts.
        p_out = p_in + p_right + p_left
        
        # Add residual connection from a previous layer if provided.
        if hidden is not None:
            p_out = hidden + p_out
            
        # Project to output dimension and apply ReLU activation.
        out = F.relu(self.conv_out(p_out))
        
        # Transpose back to (batch, length, channels) format.
        out = out.transpose(1, 2)
        
        return out, p_out

    @torch.jit.export
    def infer(self, x, buffer, buffer_index, buffer_out, hidden=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        """
        Inference pass for a single time step (streaming).
        This method is optimized for TorchScript and designed for real-time applications.

        Args:
            x (Tensor): Input tensor for a single time step.
            buffer (Tensor): A large, flat tensor used to store the layer's state (cache).
            buffer_index (Tensor): The current starting index for this layer in the buffer.
            buffer_out (Optional[Tensor]): A tensor to store the updated buffer segments.
            hidden (Optional[Tensor]): Hidden state from a previous layer for the current time step.

        Returns:
            Tuple containing the output tensor, the updated buffer, the next buffer index,
            the updated buffer_out, and the intermediate `p_out` tensor.
        """
        # Project the single time-step input.
        p_in_raw = self.conv_in(x)

        # --- Manage main input buffer for convolutions ---
        # Retrieve the cached context from the flat buffer.
        cnn_buffer = buffer[buffer_index : buffer_index + self.buffer_size].reshape(
            [1, self.hidden_dim, self.cache_size]
        )
        # Concatenate the cache with the current input to form the convolution window.
        p_in = torch.cat([cnn_buffer, p_in_raw], dim=2)
        # Store the new state (the last `cache_size` elements) for the next time step.
        buffer_out.append(p_in[:, :, -self.cache_size :].reshape(-1))
        buffer_index = buffer_index + self.buffer_size

        # --- Apply convolutions on the constructed window ---
        # Calculate left context.
        if self.left_frame > 0:
            if self.right_frame > 0:
                p_left = p_in[:, :, : -self.right_frame * self.right_dilation]
            else:
                p_left = p_in[:, :]
            p_left_out = self.conv_left(p_left)
        else:
            p_left_out = torch.tensor([0]) # Use a tensor for JIT compatibility.

        # Calculate right context.
        if self.right_frame > 0:
            p_right = p_in[:, :, self.left_frame * self.left_dilation + 1 :]
            p_right_out = self.conv_right(p_right)
        else:
            p_right_out = torch.tensor([0])

        # --- Manage buffer for the raw projected input (p_in_raw) ---
        if self.right_frame > 0:
            # This caching is necessary to correctly align `p_in_raw` with the delayed
            # context outputs `p_left_out` and `p_right_out`.
            p_in_raw_cnn_buffer = buffer[
                buffer_index : buffer_index + self.p_in_raw_buffer_size
            ].reshape([1, self.hidden_dim, self.p_in_raw_chache_size])
            p_in_raw = torch.cat([p_in_raw_cnn_buffer, p_in_raw], dim=2)
            buffer_out.append(p_in_raw[:, :, -self.p_in_raw_chache_size :].reshape(-1))
            buffer_index = buffer_index + self.p_in_raw_buffer_size
            # Get the correctly delayed `p_in_raw` for the current output time step.
            p_in_raw = p_in_raw[:, :, : -self.p_in_raw_chache_size]
        
        # Combine the components to get the intermediate output.
        p_out = p_in_raw + p_left_out + p_right_out

        # --- Manage buffer for the hidden state (if provided) ---
        if hidden is not None:
            if self.right_frame > 0:
                # Similar caching and delay logic as for `p_in_raw`.
                hidden_cnn_buffer = buffer[
                    buffer_index : buffer_index + self.hidden_buffer_size
                ].reshape([1, self.hidden_dim, self.hidden_chache_size])
                hidden = torch.cat([hidden_cnn_buffer, hidden], dim=2)
                buffer_out.append(hidden[:, :, -self.hidden_chache_size :].reshape(-1))
                buffer_index = buffer_index + self.hidden_buffer_size
                hidden = hidden[:, :, : -self.hidden_chache_size]
            # Add the (potentially delayed) hidden state.
            p_out = hidden + p_out

        # Final projection and activation.
        out = F.relu(self.conv_out(p_out))

        return out, buffer, buffer_index, buffer_out, p_out
