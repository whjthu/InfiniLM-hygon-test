from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;P=Path('/refactor/incoming/Deep-public-fixed-cohort-profile-r465/source-v1');B=Path('/refactor/candidates/Deep-public-C4-remaining-quality-CPU-v1');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();B.mkdir(exist_ok=False);q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,SDK=False,weights=False,new_compile=0,reencode=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],shards=[])
 try:
  manifest=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert H(D/'SOURCE_MANIFEST.json')==os.environ['DEEP_REMAINING_SOURCE_SHA']
  for n,h in manifest['files_SHA256'].items():assert H(D/n)==h,n
  plan=json.loads((D/'SHARD_PLAN.json').read_bytes());assert plan['HTTP']==448 and plan['questions']==224
  assert sorted(plan['excluded_pilot_ordinals']+sum([v['ordinals']for v in plan['shards']],[]))==list(range(256))
  spec=importlib.util.spec_from_file_location('original465inputs',P/'pilot_inputs.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
  for v in plan['shards']:
   j=v['shard'];builder.ORDINALS=v['ordinals'];p=B/('shard%d-private.json'%j);private=builder.build('/refactor/candidates/Deep-broad-fewshot-fullpre-quality-r450-CPU-v1/encoding.json','/refactor/incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json');private['selection']=plan['selection'];p.write_bytes((json.dumps(private,ensure_ascii=False)+'\n').encode())
   leaf=D/('shard%d'%j);env=dict(os.environ,PYTHONPATH=':'.join(map(str,[leaf,P,Path('/refactor/incoming/public-Deep-native-provider-r427/source-v3'),Path('/old/offline-deps')])));env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
   script="import json,sys;from pathlib import Path;import controller as C;from preflight import check;h=C.sha;m=json.loads((C.D/'SOURCE_MANIFEST.json').read_bytes());m['private_plan_SHA256']=h(Path(sys.argv[1]));proof=check(C.D,C.PACKAGE,m);g=C.gpu_command();a=C.client_command('"+str(v['parent'])+".recording',m,1.);assert '--jobid="+str(v['parent'])+"'in g and g[g.index('-w')+1]=="+repr(v['node'])+";assert a[a.index('--private-plan')+1]==sys.argv[1];print(json.dumps({'passed':True,'controller':str(C.__file__),'GPU_argv':g,'client_argv':a,'proof':proof}))"
   z=subprocess.run([sys.executable,'-B','-c',script,str(p)],env=env,capture_output=True,text=True,timeout=30);assert z.returncode==0,(j,z.stdout,z.stderr);control=json.loads(z.stdout)
   args=[sys.executable,'-B','-m','public_deep_cohort','--config',str(leaf/'config.json'),'--config-sha',H(leaf/'config.json'),'--source-sha',manifest['provider_source_SHA256'],'--run',v['run'],'--actual-step',q['actual_CPU'],'--gpu-start-epoch','0','--experimental-profile','deep-hc0-fixed-cohort-c4']
   z=subprocess.run(args,env=env,capture_output=True,text=True,timeout=30);assert z.returncode==0,(j,z.stdout,z.stderr);cli=json.loads(z.stdout);assert cli['source_and_schema_valid']and not cli['Native_dlopen']
   q['shards'].append(dict(shard=j,ordinals=v['ordinals'],private_path=str(p),private_SHA256=H(p),control=control,actual_nonserve_CLI=cli))
  maps=Path('/proc/self/maps').read_text();assert not any(k in maps for k in['libinfinirt','libinfiniops','libInfiniRT','libInfiniOP','_infinilm.cpython']);q.update(passed=True,source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),project_maps=[])
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 q.update(completed=True,wall_s=time.monotonic()-start);(B/'result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(dict(passed=q['passed'],actual_CPU=q['actual_CPU'],shards=len(q['shards']),error=q.get('error'))))
 if not q['passed']:raise SystemExit(1)
if __name__=='__main__':main()
