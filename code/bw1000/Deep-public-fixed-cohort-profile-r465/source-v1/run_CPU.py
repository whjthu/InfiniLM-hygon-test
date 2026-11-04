"""Proposed one changed-entry CPU: real945 metadata, typed descriptor, actualCLI; no dlopen."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;R=D.parents[2];B=Path('/refactor/candidates/Deep465-public-fixed-cohort-CPU-v1');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();B.mkdir(exist_ok=False);q=dict(completed=False,passed=False,GPU=0,Native_dlopen=False,SDK=False,weights=False,new_compile=0,reencode=False,Model_READY=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
 try:
  mf=H(D/'SOURCE_MANIFEST.json');assert mf==os.environ['DEEP465_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for n,h in m['files_SHA256'].items():assert H(D/n)==h
  env=dict(os.environ,PYTHONPATH=':'.join(map(str,[D,R/'incoming/public-Deep-native-provider-r427/source-v3',Path('/old/offline-deps')])));env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
  def run(argv):
   z=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=max(1,250-(time.monotonic()-start)));q.setdefault('commands',[]).append(dict(argv=argv,rc=z.returncode,stdout=z.stdout,stderr=z.stderr));assert z.returncode==0,z.stderr;return json.loads(z.stdout)
  q['changed_local_production_consumers']=run([sys.executable,'-B',str(D/'check_local_CPU.py')])
  q['typed_binding']=run([sys.executable,'-B','-c',"from public_deep_cohort.native_binding import binding;import json;d,r,n=binding();assert d['rows']==4 and d['capacity_per_row']==32768 and not d['standard_public_C1'];print(json.dumps({'rows':d['rows'],'profile':d['profile'],'Native_SHA256':r['library_SHA256'],'objects':len(n['link']['full']['objects']),'Native_dlopen':False}))"])
  q['actual_pilot_metadata']=run([sys.executable,'-B',str(D/'pilot_inputs.py'),'--encoding','/refactor/candidates/Deep-broad-fewshot-fullpre-quality-r450-CPU-v1/encoding.json','--gold','/refactor/incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json','--output',str(B/'pilot-private.json')])
  args=[sys.executable,'-B','-m','public_deep_cohort','--config',str(D/'config.template.json'),'--config-sha',H(D/'config.template.json'),'--source-sha',mf,'--run','CPU465.nonserve','--actual-step',q['actual_CPU'],'--gpu-start-epoch','0','--experimental-profile','deep-hc0-fixed-cohort-c4']
  q['actual_nonserve_CLI']=run(args);assert q['actual_nonserve_CLI']['source_and_schema_valid']and not q['actual_nonserve_CLI']['Native_dlopen']
  maps=Path('/proc/self/maps').read_text();assert not any(k in maps for k in['libinfinirt','libinfiniops','libInfiniRT','libInfiniOP','_infinilm.cpython'])
  q.update(passed=True,source_manifest_SHA256=mf)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 q.update(completed=True,wall_s=time.monotonic()-start);(B/'result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(passed=q['passed'],actual_CPU=q['actual_CPU'],error=q.get('error'))))
 if not q['passed']:raise SystemExit(1)
if __name__=='__main__':main()
