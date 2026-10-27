"""Actual one-card preweight imports; no model constructor or tensor weights."""
import pathlib,json,hashlib,sys,os,datetime,traceback
R=pathlib.Path('/refactor');B=R/'candidates/toolcall-Qwen06-r138';MODEL=pathlib.Path('/shared/Qwen3-0.6B');out=pathlib.Path(sys.argv[1]);assert out.is_relative_to(R/'results')and not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_only':False,'allocated_GPU_cards':1,'phase':'preweight_import','model_constructed':False,'weight_tensors_loaded':False,'HTTP_started':False}
try:
 import torch
 q['logical_visible_cards']=torch.cuda.device_count();assert q['logical_visible_cards']==1,'Slurm one-card visibility not preserved'
 p=torch.cuda.get_device_properties(0);assert p.total_memory>=63*1024**3 and('gfx936'in p.name or'gfx936'in getattr(p,'gcnArchName',''))
 q['hardware']={'name':p.name,'arch':getattr(p,'gcnArchName',None),'bytes':p.total_memory};q['free_bytes']=torch.cuda.mem_get_info(0)[0];assert q['free_bytes']>=4*1024**3
 import infinilm
 from infinilm.server.inference_server import InferenceServer
 from infinilm.server.tool_parsers import TOOL_CALL_FORMATS
 assert 'hermes'in TOOL_CALL_FORMATS
 q['server_module']=sys.modules[InferenceServer.__module__].__file__;assert pathlib.Path(q['server_module']).is_relative_to(B)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
 tool={'type':'function','function':{'name':'calculate','description':'Add integers','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}
 rendered=tok.apply_chat_template([{'role':'user','content':'Use calculate to add two integers.'}],tools=[tool],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(rendered,add_special_tokens=False)
 assert '<tool_call>'in rendered and'calculate'in rendered and len(ids)<2048
 q['template']={'token_count':len(ids),'rendered_SHA256':hashlib.sha256(rendered.encode()).hexdigest(),'enable_thinking':False}
 q['native_modules']={n:{'path':v.__file__,'SHA256':sha(v.__file__)}for n,v in list(sys.modules.items())if(n.startswith('infinilm')or n=='_infinilm')and getattr(v,'__file__',None)and v.__file__.endswith('.so')};assert q['native_modules']
 q.update(passed=True,standard_server_imported=True,hermes_parser_imported=True,tokenizer_real_rendered=True,Torch_initialized=bool(torch.cuda.is_initialized()),scope='Actual GPU import/template smoke, not model or HTTP qualification; child exits before weight load')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','logical_visible_cards','standard_server_imported','error']}));assert q['passed']
