"""Actual checkpoint ordinary-text BPE, CPU/tokenizer files only; no weights.
This adapter is for the separately frozen r181 output checker after real runs.
"""
from pathlib import Path
from xtml_codec_r143 import KimiXTMLCodec
_codec=None
def count_tokens(text):
 global _codec
 if not isinstance(text,str):raise TypeError('text must be str')
 if _codec is None:
  _codec=KimiXTMLCodec('/shared/Kimi-K3')
  if _codec.identity['tokenizer_config_SHA256']!='5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e' or _codec.identity['tiktoken_model_SHA256']!='b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103':raise ValueError('actual official tokenizer SHA mismatch')
 return len(_codec.encoding.encode(text,disallowed_special=()))
