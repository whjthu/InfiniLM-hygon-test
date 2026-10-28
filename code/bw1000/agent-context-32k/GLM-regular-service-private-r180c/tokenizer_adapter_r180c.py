"""Official CPU tokenizer for argument-content measurement, not model output."""
import os,pathlib,hashlib
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
os.environ['USE_TORCH']='0';os.environ['USE_TF']='0'
from transformers import PreTrainedTokenizerFast
_model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3')
_tokenizer=PreTrainedTokenizerFast.from_pretrained(_model,local_files_only=True)
assert hashlib.sha256(_tokenizer.get_chat_template().encode()).hexdigest()=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
def count_tokens(text):
 if not isinstance(text,str):raise TypeError('Token counter expects complete content text')
 return len(_tokenizer.encode(text,add_special_tokens=False))
