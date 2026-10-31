"""Official Rust tokenizer adapter in CPU client process; never reads weights."""
import os
from tokenizer_cpu_r173b import OfflineTokenizer
_tokenizer=None
def count_tokens(text):
 global _tokenizer
 if _tokenizer is None:_tokenizer=OfflineTokenizer(os.environ.get('DEEP_MODEL_VIEW_R3616','/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8'))
 return len(_tokenizer.encode(text,add_special_tokens=False))
