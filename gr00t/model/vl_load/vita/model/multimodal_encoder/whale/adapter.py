"""
Thin re-export shim to deduplicate Adapter implementations.

We reuse `CNNAdapter`, `LinearAdapter`, and `CNNSubsampling` from the TTS side
to keep behavior identical and avoid maintaining two copies.
"""

from vl_load.vita.model.vita_tts.adapter import (  # type: ignore
    CNNAdapter as _TTS_CNNAdapter,
    LinearAdapter as _TTS_LinearAdapter,
    CNNSubsampling as _TTS_CNNSubsampling,
)

CNNAdapter = _TTS_CNNAdapter
LinearAdapter = _TTS_LinearAdapter
CNNSubsampling = _TTS_CNNSubsampling

__all__ = ["CNNAdapter", "LinearAdapter", "CNNSubsampling"]
