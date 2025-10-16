"""
Thin re-export shim to deduplicate Subsampling implementation.

This module reuses the `Subsampling` from
`vl_load.vita.model.vita_tts.encoder.subsampling` to keep Whale encoder
behavior while removing duplicate code.
"""

from vl_load.vita.model.vita_tts.encoder.subsampling import (  # type: ignore
    BaseSubsampling as _TTS_BaseSubsampling,
    Conv2dSubsampling4 as _TTS_Conv2dSubsampling4,
    Subsampling as _TTS_Subsampling,
)

# Re-export for backward compatibility with Whale imports
BaseSubsampling = _TTS_BaseSubsampling
Conv2dSubsampling4 = _TTS_Conv2dSubsampling4
Subsampling = _TTS_Subsampling

__all__ = ["BaseSubsampling", "Conv2dSubsampling4", "Subsampling"]
