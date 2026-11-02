"""One changed-source run after the preserved pre-Native shell failure844."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-V-on-demand-Model-r404'
LABEL='GLM-V-on-demand-Model-r404-model-20261103-c';OLD_LABEL='GLM-V-on-demand-Model-r404-model-20261103-b';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication-commit',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 m=json.loads((S/'GPU_MANIFEST_c.json').read_bytes());assert sha(S/'GPU_MANIFEST_c.json')==a.manifest_sha and m['GPU_GO']is True and m['Model_GO']is True
 for n,h in m['additional_files_SHA256'].items():assert sha(S/n)==h,n
 assert m['controller_entry']=='controller/controller_v5.py'and sha(S/m['controller_entry'])==m['controller_entry_SHA256']
 spec=importlib.util.spec_from_file_location('actual_controller_v4',S/m['controller_entry']);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);C.verify(R,S,a.manifest_sha)
 shellspec=importlib.util.spec_from_file_location('actual_shell_contract_c',S/'check_shell_layout_v2.py');B=importlib.util.module_from_spec(shellspec);shellspec.loader.exec_module(B);shell=B.check(S,Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json'),'controller/controller_v5.py');assert shell['passed']
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-31','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode'],text=True,timeout=30).splitlines()if'|'in s]
 for handle,label in [('910376.847',OLD_LABEL),('910407.817',OLD_LABEL+'-keeper')]:
  matches=[r for r in rows if r[0]==handle and r[1]==label];assert len(matches)==1 and matches[0][2:4]==['FAILED','1:0']
 assert not any(r[1]in[LABEL,LABEL+'-keeper']for r in rows),'Follow existing changed-source label, no resubmission'
 prior=R/'results'/(OLD_LABEL+'-watch.json');assert sha(prior)=='ea896106ceed543e9481abead61ce9eafac29cc27fd98f8102f1603a270f7792';w=json.loads(prior.read_bytes());assert w['completed']and not w['passed']and w['actual_GPU']=='910376.847'and not w['forced_cleanup']
 assert len(w['post_after_failure'])==2 and all(not r['kfd_pids']and not r['matching_component_processes']and r['ports_bindable']=={'43843':True}for r in w['post_after_failure'])
 intent=R/'results/GLM-r404-Model-intent-c.json';watch=R/'results'/(LABEL+'-watch.json');client=R/'logs'/(LABEL+'-keeper-client.log');output=R/'logs'/(LABEL+'-keeper-%J.log')
 candidates=[intent,watch,client,R/'control'/(LABEL+'-GO.json'),R/'control'/(LABEL+'-cancel.json'),R/'logs'/(LABEL+'-srun-client.log')];assert not any(v.exists()for v in candidates)and not list((R/'logs').glob(LABEL+'*'))
 parent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);assert 'JobState=RUNNING'in parent and'a14r2n20'in parent
 argv=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w','a14r2n20','-t','01:40:00','--job-name='+LABEL+'-keeper','--output='+str(output),'--export=ALL','python3','-B','-u',str(S/m['controller_entry']),'--publication-commit',a.publication_commit,'--manifest-sha',a.manifest_sha]
 data=dict(changed_source_after_GPU847_preengine_failure=True,prior_watch_SHA256=sha(prior),GPU_attempt=3,Model_load_attempt=1,publication_commit=a.publication_commit,GPU_manifest_SHA256=a.manifest_sha,created_unix=time.time(),attempt=1,no_retry=True,keeper_argv=argv,shell_contract=shell)
 if a.preflight_only:print(json.dumps(dict(preflight_passed=True,actual_launcher_CLI=True,Slurm_submitted=0,intent_created=False,log_created=False,GPU_submitted=False,**data)));return
 with intent.open('x')as out:json.dump(data,out,indent=2)
 with client.open('x')as log:proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 data.update(keeper_srun_PID=proc.pid,keeper_label=LABEL+'-keeper',GPU_label=LABEL);tmp=intent.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(intent);print(json.dumps(data))
if __name__=='__main__':main()
