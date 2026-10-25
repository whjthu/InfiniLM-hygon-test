import pathlib,json,hashlib,os,ast,subprocess,traceback,sys,importlib
r=pathlib.Path('/refactor');inc=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'retokenized':False,'weights_loaded':False}
try:
 parent=r/'incoming/GLM-real-concurrency-r9132/CPU-result.json';assert sha(parent)=='f82d9f9a0aa2c685da20eaeb669882f666a978123f8536b041cd8a4152f9c540';q['parent_CPU336_SHA256']=sha(parent);old=json.loads(parent.read_text());assert old['completed'];program=r/'results/GLM-actual-concurrency-tokenized-r9132.json';assert sha(program)==old['program_SHA256']=='da6aadfa32701ef8684bba489dd3234151c34933f8df1736c385ab70c3539cdc';q['program_SHA256']=sha(program);q['program_path']=str(program)
 m=inc/'GLM-CROSSREQUEST-Model-source-manifest-r9059.json';assert sha(m)==sha(r/'incoming/GLM-CROSSREQUEST-Model-source-manifest-r9059.json');q['same_directory_648_manifest_SHA256']=sha(m)
 modules=['glm_actual_concurrency_probe_r9132b','glm_crossrequest_control_r9059','glm_crossrequest_resident_base_r9059','quality_metrics_r9132b','schedule_r9132b'];actual={}
 for name in modules:
  module=importlib.import_module(name);p=pathlib.Path(module.__file__);assert p.parent==inc,(name,str(p));actual[name]={'path':str(p),'SHA256':sha(p)}
 import torch
 assert not torch.cuda.is_initialized();q['CUDA_initialized']=False;q['actual_import_chain']=actual;q['relative_manifest_read_by_actual_control_import']=True
 from glm_crossrequest_resident_base_r9059 import BANK_BYTES,FLOOR
 assert BANK_BYTES==160*1024**2 and FLOOR==512*1024**2
 from schedule_r9132b import selftest,prefix_chunks,microbatches
 q['schedule_CPU']=selftest();prompts=[list(range(638-i))for i in range(32)];seen={i:[]for i in range(32)}
 for rows,tokens,pasts in prefix_chunks(prompts,list(range(32)),8,4096):
  assert sum(map(len,tokens))<=4096
  for i,t,p in zip(rows,tokens,pasts):assert p==len(seen[i]);seen[i]+=t
 assert all(seen[i]==prompts[i]for i in range(32));q['actual4096split_reassembly']=True
 from quality_metrics_r9132b import selftest as quality_selftest
 q['quality_CPU']=quality_selftest()
 for p in inc.glob('*.py'):ast.parse(p.read_text())
 for p in inc.glob('*.sh'):assert b'\r'not in p.read_bytes();subprocess.run(['bash','-n',str(p)],check=True)
 meta=old['parent648_preflight']['model_library_metadata'];library=pathlib.Path(meta['library_actual_path']);assert sha(library)==meta['library_SHA256']=='fbd07490dd9887868e4a6f5303170969dfa188e9cbcb39f7dca708b3a303741a';q['parent648_preflight']=old['parent648_preflight'];q['MMLU_kept']=old['MMLU_kept'];q['MMLU_excluded']=old['MMLU_excluded'];q['service_distinct']=old['service_distinct'];q['service_input_range']=old['service_input_range'];q['service_excluded']=old['service_excluded'];q['source_SHA256']={p.name:sha(p)for p in inc.iterdir()if p.is_file()};q['completed']=True
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(inc/'CPU-result.json').write_text(json.dumps(q,indent=2));print(json.dumps({k:q.get(k)for k in ['completed','producer','error','retokenized','weights_loaded','program_SHA256','same_directory_648_manifest_SHA256']}));assert q['completed']
