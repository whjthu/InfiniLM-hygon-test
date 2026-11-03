"""Source-only CPU CLI or one authorized full-preflight keeper submission."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,subprocess,time,sys
R=Path('/refactor')if Path('/refactor').is_dir()else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=Path(__file__).resolve().parent;LABEL='GLM-NoV-16card-W4-pair-r414-pair-20261103-a';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(S))
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--manifest-file',default='GPU_MANIFEST.json',choices=['GPU_MANIFEST.json','GPU_MANIFEST.template.json']);p.add_argument('--source-preflight-only',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 from verify_source import verify_source
 m=verify_source(R,S,a.manifest_sha,a.manifest_file)
 for n,h in m['additional_files_SHA256'].items():assert sha(S/n)==h,n
 from check_shell_layout import check
 config=Path('/old/models/ZhipuAI-GLM-5.3/config.json')
 if not config.is_file():config=Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json')
 shell=check(S,config,'controller/controller_v2.py')
 assert shell['passed']and all(r['actual_same_typed_layout']['config_SHA256']==m['config_SHA256']for r in shell['roles'])
 if a.source_preflight_only:
  print(json.dumps(dict(source_preflight_passed=True,shell_contract=shell,Slurm=0,GPU=0,Native_dlopen=False,intent_created=False)));return
 assert a.manifest_file=='GPU_MANIFEST.json'and m['GPU_GO']is True and m['Model_GO']is True
 spec=importlib.util.spec_from_file_location('final_private_controller',S/'controller/controller_v2.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);C.verify(R,S,a.manifest_sha)
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-31','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode'],text=True,timeout=30).splitlines()if'|'in s]
 assert not any(r[1]in[LABEL,LABEL+'-keeper']for r in rows),'Follow exact original label; never resubmit'
 for handle in ['910376.850','910407.822']:
  matches=[r for r in rows if r[0]==handle];assert len(matches)==1 and matches[0][2:4]==['COMPLETED','0:0']
 intent=R/'results/GLM-r414-Model-intent.json';watch=R/'results'/(LABEL+'-watch.json');client=R/'logs'/(LABEL+'-keeper-client.log')
 assert not any(p.exists()for p in[intent,watch,client,R/'control'/(LABEL+'-GO.json'),R/'control'/(LABEL+'-cancel.json')])and not list((R/'logs').glob(LABEL+'*'))
 parent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert'JobState=RUNNING'in parent and'a14r2n20'in parent
 argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','01:40:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'--export=ALL','python3','-B','-u',str(S/'controller/controller_v2.py'),'--publication-commit',a.publication_commit,'--manifest-sha',a.manifest_sha]
 data=dict(publication_commit=a.publication_commit,GPU_manifest_SHA256=a.manifest_sha,created_unix=time.time(),attempt=1,no_retry_reload=True,keeper_argv=argv,windows=m['windows'],HTTP=6,Native=6,shell_contract=shell)
 if a.preflight_only:print(json.dumps(dict(preflight_passed=True,intent_created=False,Slurm=0,GPU=0,**data)));return
 with intent.open('x')as f:json.dump(data,f,indent=2)
 with client.open('x')as log:proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 data.update(srun_PID=proc.pid,GPU_label=LABEL,keeper_label=LABEL+'-keeper');tmp=intent.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(intent);print(json.dumps(data))
if __name__=='__main__':main()
