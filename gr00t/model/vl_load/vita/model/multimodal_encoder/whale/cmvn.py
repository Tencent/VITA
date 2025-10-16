"""
Thin re-export shim to deduplicate CMVN implementation.

This module reuses the vetted implementation in
`vl_load.vita.model.vita_tts.encoder.cmvn` to avoid code duplication while
preserving the public API expected by Whale modules.
"""

from vl_load.vita.model.vita_tts.encoder.cmvn import (  # type: ignore
    GlobalCMVN as _TTS_GlobalCMVN,
    load_cmvn as _tts_load_cmvn,
)

# Re-export with the same names used in Whale code
GlobalCMVN = _TTS_GlobalCMVN


def load_cmvn(filename, is_json):
    return _tts_load_cmvn(filename, is_json)


__all__ = ["GlobalCMVN", "load_cmvn"]
