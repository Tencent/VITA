import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

"""
Note on duplication:
This module mirrors `clip/clip_encoder.py` at a high level to keep a consistent
vision-tower interface. We keep behavior identical but with an InternViT
backbone and a pixel_shuffle step. Minimal renames and comments reduce
duplication without changing logic.
"""

from .modeling_intern_vit import InternVisionModel


class InternViTVisionTower(nn.Module):
    """
    Vision tower using InternViT as the backbone encoder.
    Processes images and extracts visual features with pixel shuffle downsampling.
    """
    def __init__(self, vision_tower, args, delay_load=False):
        """
        Initialize the InternViT vision tower.
        
        Args:
            vision_tower: Name or path of the vision tower model
            args: Configuration arguments
            delay_load: If True, delay loading the model weights
        """
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = -1  # Use the last hidden layer
        self.scale_pix_shuffle = 0.5  # Pixel shuffle scale factor

        if not delay_load:
            self.load_model()
        else:
            # Load config only without weights
            self.cfg_only = AutoConfig.from_pretrained(
                self.vision_tower_name, trust_remote_code=True
            )

    def load_model(self):
        """Load the vision tower model and image processor."""
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = InternVisionModel.from_pretrained(
            self.vision_tower_name, trust_remote_code=True
        )
        # Freeze vision tower parameters
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, vision_outs):
        """
        Select features from a specific layer of vision outputs.
        
        Args:
            vision_outs: Vision model outputs containing hidden states
        
        Returns:
            Selected features excluding the CLS token
        """
        feats = vision_outs.hidden_states[self.select_layer]
        feats = feats[:, 1:]  # Remove CLS token
        return feats

    def pixel_shuffle(self, feat_map, scale_factor=0.5):
        """
        Apply pixel shuffle to reduce spatial dimensions and increase channels.
        
        Args:
            feat_map: Feature map tensor (N, W, H, C)
            scale_factor: Scale factor for spatial dimension reduction
        
        Returns:
            Shuffled feature map with reduced spatial size
        """
        n, w, h, c = feat_map.size()
        y = feat_map.view(n, w, int(h * scale_factor), int(c / scale_factor))
        y = y.permute(0, 2, 1, 3).contiguous()
        y = y.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        y = y.permute(0, 2, 1, 3).contiguous()
        return y

    #@torch.no_grad()
    def forward(self, images):
        """
        Forward pass to extract visual features from images.
        
        Args:
            images: Single image tensor or list of image tensors
        
        Returns:
            Processed image features after pixel shuffle
        """
        # Handle both single tensor and list of tensors
        if type(images) is list:
            image_features = []
            for image in images:
                # Process each image individually
                vision_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                feature = self.feature_select(vision_out).to(image.dtype)
                image_features.append(feature)
        else:
            # Process batched images
            vision_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
            )
            image_features = self.feature_select(vision_outs).to(images.dtype)
        
        # Reshape to 2D grid (h, w) and apply pixel shuffle
        h = w = int(image_features.shape[1] ** 0.5)
        assert image_features.shape[1] == h * w
        image_features = image_features.reshape(image_features.shape[0], h, w, -1)
        image_features = self.pixel_shuffle(image_features * self.scale_pix_shuffle)
        # Flatten back to sequence
        image_features = image_features.reshape(
            image_features.shape[0], -1, image_features.shape[-1]
        )

        return image_features

    @property
    def dummy_feature(self):
        """Generate a dummy feature tensor for initialization."""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """Return the data type of the vision tower."""
        return self.vision_tower.dtype

    @property
    def device(self):
        """Return the device of the vision tower."""
        return self.vision_tower.device

    @property
    def config(self):
        """Return the configuration of the vision tower."""
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """Return the hidden size after pixel shuffle adjustment."""
        return self.config.hidden_size * (int(1 / self.scale_pix_shuffle) ** 2)

    @property
    def num_patches(self):
        """Return the total number of patches in the image."""
        return (self.config.image_size // self.config.patch_size) ** 2
