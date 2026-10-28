"""Actual offline tokenizer.json backend only; no Transformers/Torch/Native/weights."""
import pathlib,sys
class OfflineTokenizer:
 def __init__(self,model):
  assert 'torch'not in sys.modules and 'transformers'not in sys.modules,'CPU tokenizer entry requires no Torch/Transformers import'
  from tokenizers import Tokenizer
  assert 'torch'not in sys.modules and 'transformers'not in sys.modules,'tokenizers unexpectedly imported Torch/Transformers'
  self.path=pathlib.Path(model)/'tokenizer.json';self.backend=Tokenizer.from_file(str(self.path))
  assert 'torch'not in sys.modules and 'transformers'not in sys.modules,'tokenizer.json backend import side effect'
 def encode(self,text,add_special_tokens=False):
  assert add_special_tokens is False,'Official prepared prompt already contains special tokens'
  return list(self.backend.encode(text,add_special_tokens=False).ids)
 def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
  assert clean_up_tokenization_spaces is False
  return self.backend.decode(list(ids),skip_special_tokens=skip_special_tokens)
