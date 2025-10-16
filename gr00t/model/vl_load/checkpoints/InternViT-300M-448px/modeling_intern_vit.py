"""
Thin re-export to avoid code duplication.

This file was a local copy of the InternViT vision model. We now reuse the
authoritative implementation under `vl_load.vita.model.multimodal_encoder.internvit`.
"""

from vl_load.vita.model.multimodal_encoder.internvit.modeling_intern_vit import *  # type: ignore
