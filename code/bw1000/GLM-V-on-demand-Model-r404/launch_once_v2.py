"""Host-only unique keeper launch after explicit final Model/GPU authorization."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-V-on-demand-Model-r404'
LABEL='GLM-V-on-demand-Model-r404-model-20261103-a';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 m=json.loads((S/'GPU_MANIFEST.json').read_bytes());assert sha(S/'GPU_MANIFEST.json')==a.manifest_sha and m['GPU_GO']is True and m['Model_GO']is True
 assert sha(S/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256']and sha(S/'owner/SOURCE_MANIFEST.json')==m['owner_manifest_SHA256']
 for n,h in json.loads((S/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(S/n)==h,n
 for n,h in m['additional_files_SHA256'].items():assert sha(S/n)==h,n
 assert m['controller_entry']=='controller/controller_v3.py'and sha(S/m['controller_entry'])==m['controller_entry_SHA256']
 receipt=R/'candidates/GLM-V-on-demand-Model-r404-CPU-v3/CPU-result.json';assert sha(receipt)==m['CPU_receipt_SHA256'];cpu=json.loads(receipt.read_bytes());assert cpu['completed']and cpu['passed']and cpu['GPU']==cpu['Model']==0 and not cpu['Native_dlopen']
 rows=subprocess.check_output(['sacct','-S','2026-10-31','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%160,State'],text=True,timeout=30)
 parsed=[x.split('|')for x in rows.splitlines()if'|'in x]
 old=[x for x in parsed if x[1]==LABEL+'-keeper'];assert len(old)==1 and old[0][0]=='910407.814'and old[0][2]=='FAILED'
 assert not any(x[1]in[LABEL,LABEL+'-keeper-v2']for x in parsed),'Follow original GPU/newkeeper labels; no replay'
 prior=R/'results/GLM-r404-Model-intent.json';prior_bytes=prior.read_bytes();previous=json.loads(prior_bytes)
 assert previous['publication_commit']=='d0f7b1ebc7b634c16cb8f9eabdd4aae0e53e0504'and previous['GPU_manifest_SHA256']=='60c3b7726756b78828831742497651556a31691424b7203077e24f8669a67874'

 intent=R/'results/GLM-r404-Model-intent-v2.json';watch=R/'results'/(LABEL+'-watch.json');assert not intent.exists()and not watch.exists()
 parent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert 'JobState=RUNNING'in parent and'a14r2n20'in parent
 argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','01:40:00','--job-name='+LABEL+'-keeper-v2','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'--export=ALL','python3','-B','-u',str(S/m['controller_entry']),'--publication-commit',a.publication_commit,'--manifest-sha',a.manifest_sha]
 data=dict(recovery_of_preGPU_keeper='910407.814',prior_intent_SHA256=hashlib.sha256(prior_bytes).hexdigest(),GPU_attempt=1,keeper_attempt=2,publication_commit=a.publication_commit,GPU_manifest_SHA256=a.manifest_sha,created_unix=time.time(),attempt=1,no_retry=True,keeper_argv=argv)
 with intent.open('x')as f:json.dump(data,f,indent=2)
 with(R/'logs'/(LABEL+'-keeper-client.log')).open('x')as log:proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 data.update(keeper_srun_PID=proc.pid,keeper_label=LABEL+'-keeper-v2',GPU_label=LABEL);tmp=intent.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(intent)
 print(json.dumps(data))
if __name__=='__main__':main()
