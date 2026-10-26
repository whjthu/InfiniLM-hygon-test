import pathlib,json,hashlib,os,ast,subprocess,traceback,importlib
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'retokenized':False,'weights_loaded':False}
try:
 prior=r/'incoming/GLM-real-concurrency-r9132b/CPU-result.json';assert sha(prior)=='ef6f257fb04f53aa0976f67dc4e1cc447fa63d44a3a3056c3b4471544c976c7f';old=json.loads(prior.read_text());program=r/'results/GLM-B32-tokenized-r9183.json';prepared=json.loads((inc/'tokenizer-result.json').read_text());assert prepared['completed']and sha(program)==prepared['program_SHA256'];q.update(program_SHA256=sha(program),program_path=str(program),parent_CPU338_SHA256=sha(prior))
 modules=['glm_ragged_probe_r9183','preflight_r9183','preengine_r9183','glm_crossrequest_control_r9059','glm_crossrequest_resident_base_r9059','quality_metrics_r9132b','schedule_r9132b'];actual={}
 for name in modules:
  mod=importlib.import_module(name);f=pathlib.Path(mod.__file__);assert f.parent==inc,(name,str(f));actual[name]={'path':str(f),'SHA256':sha(f)}
 import torch
 assert not torch.cuda.is_initialized();q['CUDA_initialized']=False;q['actual_import_chain']=actual;from preflight_r9183 import validate;q['private_preflight']=validate();meta=q['private_preflight']['model_library_metadata'];assert meta['library_SHA256']==json.loads((r/'candidates/GLM-B32-Model-r9183/CPU-build.json').read_text())['library_SHA256']
 from glm_crossrequest_resident_base_r9059 import BANK_BYTES,FLOOR
 assert BANK_BYTES==160*1024**2 and FLOOR==512*1024**2
 from schedule_r9132b import selftest,prefix_chunks
 q['schedule_CPU']=selftest();prompts=[list(range(638-i))for i in range(32)];seen={i:[]for i in range(32)}
 for rows,tokens,pasts in prefix_chunks(prompts,list(range(32)),8,4096):
  assert sum(map(len,tokens))<=4096
  for i,t,p in zip(rows,tokens,pasts):assert p==len(seen[i]);seen[i]+=t
 assert all(seen[i]==prompts[i]for i in range(32));q['actual4096split_reassembly']=True
 from quality_metrics_r9132b import selftest as quality_selftest
 q['quality_CPU']=quality_selftest();source=inc/'slots_contract.cpp';header=r/'candidates/glm-M16-r8861/tree/candidates/glm-PP2-B32-r8843/tree/candidates/glm-prefill-prefix-r8703/src/glm_model_resident_policy_r5640.hpp';assert sha(header)=='80098ca2c6805d4fb4afe1104e2a4da65c1ceb8f98b58556908a5545eadae40a';binary=inc/'slots_contract';z=subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(header.parent),str(source),'-o',str(binary)],capture_output=True,text=True,timeout=30);assert z.returncode==0,z.stderr;q['actualSlotLifecycle_CPU']=json.loads(subprocess.check_output([str(binary)],text=True))
 for f in inc.glob('*.py'):ast.parse(f.read_text())
 for f in inc.glob('*.sh'):assert bytes([13])not in f.read_bytes();subprocess.run(['bash','-n',str(f)],check=True)
 q.update(MMLU_kept=0,MMLU_excluded=[],service_distinct=old['service_distinct'],service_input_range=prepared['service_input_range'],service_excluded=prepared['excluded'],completed=True);q['source_SHA256']={f.name:sha(f)for f in inc.iterdir()if f.is_file()and f.suffix in ['.py','.sh','.cpp','.json']}
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
(inc/'CPU-result.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','error','program_SHA256','CUDA_initialized']}));assert q['completed']
