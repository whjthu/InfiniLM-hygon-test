"""Prepare real long input files on CPU; no model-generated turns are invented."""
import pathlib,json,hashlib,importlib.util
from tokenizer_cpu_r173b import OfflineTokenizer
from chat_prepare_r173 import prepare
D=pathlib.Path(__file__).resolve().parent
def regular_module():
 p=D/'client_regular_agent.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='51218ed03190ea87b63cb06cf6c5883aae5d327a7db878109fd34c5ce4bde194';spec=importlib.util.spec_from_file_location('frozen_regular_r178',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def parity_inputs(profile_dir,tokenizer):
 profiles=json.loads((pathlib.Path(profile_dir)/'profiles.json').read_bytes());rows=[]
 for p in profiles:
  path=pathlib.Path(p['input_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==p['input_SHA256'];text=json.dumps(json.loads(path.read_bytes()),ensure_ascii=False,separators=(',',':'))
  rows.append((p['label']+'-real-tool-file',text))
  messages=[{'role':'system','content':'Use tools and preserve their actual results.'},{'role':'user','content':'Read input.json and retain every complete tool result before continuing.'}]
  prompt,_=prepare(messages,{},True,{'enable_thinking':p['thinking']},tokenizer);rows.append((p['label']+'-official-first-prompt',prompt))
 rows.append(('official-special-tokens','<｜begin▁of▁sentence｜><｜User｜>Unicode中文 4096/8192<｜Assistant｜><think>｜DSML｜'))
 return rows
def fingerprint(profile_dir,tokenizer):
 return [dict(label=label,text_SHA256=hashlib.sha256(text.encode()).hexdigest(),tokens=len(ids),ids_SHA256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())for label,text in parity_inputs(profile_dir,tokenizer)for ids in [list(tokenizer.encode(text,add_special_tokens=False))]]
def prepare_actual(model,output):
 tokenizer=OfflineTokenizer(model);module=regular_module();profiles=module.prepare_profiles(lambda s:len(tokenizer.encode(s,add_special_tokens=False)),pathlib.Path(output))
 return dict(scope='CPU actual checkpoint Rust tokenizer sizing only; not actual Model/tool/HTTP qualification',profiles=profiles,parity=fingerprint(output,tokenizer),tokenizer_json_SHA256=hashlib.sha256((pathlib.Path(model)/'tokenizer.json').read_bytes()).hexdigest(),context_capacity=32768,default_output=4096,thinking_output=8192)
