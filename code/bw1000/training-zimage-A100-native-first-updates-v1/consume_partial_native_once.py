"""Independently consume real native25/firstupdate and a failed resume gate."""
from pathlib import Path
import ast,hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
intent=json.loads((D/'launch-intent.json').read_bytes());claim=intent['claim'];request=intent['request_id']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
t=json.loads(subprocess.check_output(git+['show',claim+':coordination/tasks/'+D.name+'.json']))
files={};sources={}
for name,pin in t['source_SHA256'].items():
 raw=subprocess.check_output(git+['show',claim+':'+t['source']+'/'+name]);assert hashlib.sha256(raw).hexdigest()==pin;sources[name]=raw
for target,name in t['remote_files'].items():files[target]=dict(bytes=len(sources[name]),SHA256=hashlib.sha256(sources[name]).hexdigest())
files['src/BINDING.json']=t['private_binding_pin']
s=sources['read_original_terminal.py'].decode()
changes={
 "assert o['passed'] and o['returncode']==0 and o['owned_group_reaped'] and o['stop_reason'] is None":"assert o['passed'] is False and o['returncode']==1 and o['owned_group_reaped'] and o['stop_reason'] is None",
 "assert r['passed'] and r['continuous_updates']==25 and r['resumed_updates']==5 and r['actual_GPU_worker_processes']==2":"assert r['passed'] is False and r['stage']=='independent-private-checkpoint-comparison' and len(r['phases'])==2",
 "fm['framework_commit']==r['framework_commit']":"fm['framework_commit']=='b570fc14907dd7fafd53dd739a39c19795a1cc4a'",
 "cp25={};checkpoint_rows=[]":"cp25={};checkpoint_rows=[];frozen_by_mode={}",
 "before=small(mode+'/frozen-before.json');after=small(mode+'/frozen-after.json');assert before==after and len(before)==value['frozen_tensors'] and len(before)>=521":"before=small(mode+'/frozen-before.json');after=small(mode+'/frozen-after.json');assert before==after and len(before)==value['frozen_tensors'] and len(before)>=521;frozen_by_mode[mode]=before",
 "assert parameter_difference<=1e-6 and common.tree_sha256(a['state'])==common.tree_sha256(b['state']) and common.tree_sha256(a['scheduler'])==common.tree_sha256(b['scheduler']) and common.tree_sha256(a['rng'])==common.tree_sha256(b['rng'])":"""state_equal=common.tree_sha256(a['state'])==common.tree_sha256(b['state'])
 scheduler_equal=common.tree_sha256(a['scheduler'])==common.tree_sha256(b['scheduler']);rng_equal=common.tree_sha256(a['rng'])==common.tree_sha256(b['rng'])
 metricsA={x['global_step']:x for x in phases['continuous']['result']['step_metrics']};metricsB={x['global_step']:x for x in phases['resume20']['result']['step_metrics']}
 step_differences=[dict(global_step=i,continuous_loss=metricsA[i]['loss'],resumed_loss=metricsB[i]['loss'],loss_abs=abs(metricsA[i]['loss']-metricsB[i]['loss']),
  continuous_grad_norm=metricsA[i]['grad_norm'],resumed_grad_norm=metricsB[i]['grad_norm']) for i in range(21,26)]
 frozen_mismatch=[n for n in frozen_by_mode['continuous'] if frozen_by_mode['continuous'].get(n)!=frozen_by_mode['resume20'].get(n)]""",
 "comparison=small('resume-comparison.json');assert comparison['passed'] and comparison['parameter_max_abs']==parameter_difference":"comparison=small('resume-comparison.json');assert comparison['passed'] is False and comparison['parameter_max_abs']==parameter_difference and parameter_difference>1e-6",
 "complete_resume_state_equal=True)":"complete_resume_state_equal=state_equal,resume_acceptance_passed=False,original_resume_atol=1e-6,step_differences=step_differences,scheduler_equal=scheduler_equal,RNG_equal=rng_equal,frozen_cross_process_mismatch_count=len(frozen_mismatch),frozen_cross_process_mismatch_names=frozen_mismatch[:20])",
 "claim=CLAIM,request_id=REQUEST,artifacts=artifacts,source_files=len(EXPECTED_FILES),new_CPU_or_GPU_invocations=0,":"claim=CLAIM,request_id=REQUEST,artifacts=artifacts,original_run_passed=False,original_returncode=1,original_failure_preserved=True,resume_acceptance_passed=False,native_continuous25_and_first_capture_qualified=failure is None,source_files=len(EXPECTED_FILES),new_CPU_or_GPU_invocations=0,"
}
for old,new in changes.items():assert s.count(old)==1,old;s=s.replace(old,new)
ast.parse(s);(D/'read_partial_original_terminal.py').write_bytes(s.encode())
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nEXPECTED_FILES='+repr(files)+'\nGPU_UUID='+repr(t['GPU_UUID'])+'\nGPU_PCI='+repr(t['GPU_PCI'])+'\n').encode()+s.encode()
started=time.monotonic();r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','/home/whj/nvidia-baseline/env-zimage/bin/python -B -u -'],input=payload,capture_output=True,timeout=270)
(D/'partial-reader.stdout.private.json').write_bytes(r.stdout);(D/'partial-reader.stderr.private.txt').write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
with (D/'partial-reader-result.json').open('x') as f:json.dump(dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-started,reader_source_SHA256=hashlib.sha256(s.encode()).hexdigest(),
 stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),value=v,new_GPU_invocations=0),f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,reader_passed=v.get('passed') if v else None,failure=v.get('failure') if v else None,
 original_run_passed=False,resume_acceptance_passed=False,native_continuous25_and_first_capture_qualified=v.get('native_continuous25_and_first_capture_qualified') if v else None,
 summary=v.get('summary') if v else None,post=v.get('post') if v else None,new_GPU_invocations=0)))
