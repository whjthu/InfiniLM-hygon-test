"""CPU-only actual package/normal CLI consumer; no encode or Native execution."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
from build_overlay import build
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.monotonic();d=Path(__file__).resolve().parent;m=json.loads((d/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h
 claim=json.loads((d/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['source_manifest_SHA256']==sha(d/'SOURCE_MANIFEST.json')and claim['CPU_GO']is True
 out=Path(claim['output']);out.mkdir(parents=True,exist_ok=False);q=dict(completed=False,passed=False,GPU=False,Native=False,dlopen=False,weights=False,new_encode=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(d/'SOURCE_MANIFEST.json'))
 try:
  parent=Path('/refactor/incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1');clock=Path('/refactor/incoming/public-token-clock-r480');meta=build(parent,clock,out/'source',m['parent_r454_SHA256'],m['parent_r480_SHA256']);root=out/'source'
  config=json.loads((root/'config.template.json').read_bytes());config.update(bounded_queue_enabled=True,physical_pending_capacity=2,queue_wait_seconds=5.,token_clock_enabled=True,execute_authorized=False)
  p=out/'config.json';p.write_text(json.dumps(config,indent=2));env=dict(os.environ);env['PYTHONPATH']=str(root)+':/refactor/incoming/public-Deep-native-provider-r427/source-v3:'+env.get('PYTHONPATH','')
  args=[sys.executable,'-B','-m','public_deep_model','--config',str(p),'--config-sha',sha(p),'--source-sha',sha(root/'SOURCE_MANIFEST.json'),'--run','CPU-only','--actual-step',q['actual_step'],'--gpu-start-epoch','0']
  r=subprocess.run(args,env=env,capture_output=True,text=True,timeout=max(1,250-(time.monotonic()-start)));(out/'CLI.stdout').write_text(r.stdout);(out/'CLI.stderr').write_text(r.stderr);assert r.returncode==0,r.stderr[-2000:];report=json.loads(r.stdout.strip());assert report['Native_dlopen']is False and report['GPU']is False and not report['Model_READY']
  q.update(passed=True,normal_CLI_nonserve=True,derived_manifest_SHA256=sha(root/'SOURCE_MANIFEST.json'),CLI_stdout_SHA256=sha(out/'CLI.stdout'),queue_source_SHA256=sha(root/'public_deep_model/queue_policy.py'),same_Native_math=True)
 except BaseException as error:q['error']=repr(error);raise
 finally:
  q.update(completed=True,elapsed=time.monotonic()-start);(out/'result.json').write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':main()
