"""One CPUkeeper, which alone submits one owned GPUstep after freshidle."""
from pathlib import Path
import json,hashlib,subprocess,os,re,argparse
from controller_gate_v2 import clean_env,LABEL,NAME
KEEPER=LABEL+'-keeper'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keeper_command(root,go):
 root=Path(root);return ['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=2048M','-w','a14r2n20','-t','00:15:00','--job-name='+KEEPER,'--output='+str(root/'logs'/(KEEPER+'-%J.log')),'--export=ALL','python3','-B','-u',str(root/'incoming'/NAME/'controller_gate_v2.py'),'--execution-go',str(go)]
def start_keeper(root,go,popen=None):
 args=keeper_command(root,go)
 with(Path(root)/'logs'/(KEEPER+'-srun.log')).open('x')as f:p=(popen or subprocess.Popen)(args,env=clean_env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,args
def launch(root,publication,expected_manifest):
 root=Path(root);d=root/'incoming'/NAME;mf=d/'GPU_SOURCE_MANIFEST.json'
 assert re.fullmatch('[0-9a-f]{40}',publication)and publication!='0'*40 and sha(mf)==expected_manifest
 m=json.loads(mf.read_bytes());assert m['GPU_GO']is True and m['Model_GO']is False and not m['final_root_read_pending']
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 assert sha(root/m['CPU_receipt'].removeprefix('/refactor/'))==m['CPU_receipt_SHA256']
 go=root/'control/GLM400-device-v1-execution-go.json';intent=root/'results/GLM400-device-v1-intent.json'
 assert not go.exists()and not intent.exists()and not(root/'results/GLM400-watch-v1.json').exists()
 rows=subprocess.check_output(['sacct','-S','2026-10-31','-j','910371,910407','-n','-P','-o','JobIDRaw,JobName%160,State'],text=True,timeout=25)
 assert not any(x.split('|')[1]in[LABEL,KEEPER]for x in rows.splitlines()if'|'in x),'Original label alreadyexists; no replay'
 parent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert'UserId=aclsylqidf('in parent and'JobState=RUNNING'in parent and'a14r2n20'in parent
 execution=dict(execute_once=True,publication_commit=publication,GPU_source_manifest_SHA256=expected_manifest,Model_GO=False)
 with go.open('x')as f:json.dump(execution,f,indent=2)
 with intent.open('x')as f:json.dump(dict(**execution,GPU_cards=8,GPU_max_s=600,keeper_max_s=900,no_retry=True,CPU_label=KEEPER,GPU_label=LABEL),f,indent=2)
 p,args=start_keeper(root,go)
 return dict(submitted=True,publication_commit=publication,GPU_source_manifest_SHA256=expected_manifest,keeper_PID=p.pid,keeper_label=KEEPER,GPU_label=LABEL,argv=args,Model_GO=False,original_intent=str(intent))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--source-sha',required=True);a=p.parse_args();print(json.dumps(launch(Path.home()/'hygon-inference-refactor-20261014',a.publication,a.source_sha)))
