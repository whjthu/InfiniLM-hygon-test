"""Actual controller.main recording; Slurm/process/remote-source leaves mocked, no submission."""
from pathlib import Path
from unittest import mock
import sys,json,copy,tempfile,types
import controller_397 as c
D=Path(__file__).resolve().parent;g=json.loads((D/'execution-GO.template.json').read_bytes());g.update(GPU_authorized=True,root_consumed_final_GPU_scope=True,publication_commit='1'*40,controller_manifest_SHA256='0'*64)
c.validate_scope(g);checks=[]
for key,value in [('GPU_authorized',False),('root_consumed_final_GPU_scope',False),('cards',16),('cards',True),('max_HTTP',1),('max_Native_sequences',5),('max_forwards',369),('teacher_tokens',84),('output_decisions',85),('prefill_forwards_each',8),('decode_forwards_each',82),('GPU_max_s',1801),('Native_max_s',1800),('worker_max_s',1800),('CPU_keeper_max_s',2101),('GPU_node','a02r4n05'),('source_manifest_SHA256','0'*64),('CPU_whole_receipt_SHA256','0'*64),('HC_modes',[1]*4),('observer_enabled',True),('Native_SHA256','0'*64)]:
 bad=copy.deepcopy(g);bad[key]=value
 try:c.validate_scope(bad)
 except(AssertionError,PermissionError):pass
 else:raise AssertionError('Invalid scope accepted: '+key)
 checks.append(dict(case=key,passed=True))
for scenario in ['complete','GPU_failed','observation_lost_then_complete','driver_passed_false']:
 with tempfile.TemporaryDirectory(prefix='Deep397-controller-CPU-')as td:
  root=Path(td);(root/'results').mkdir();(root/'logs').mkdir();go=root/'go.json';go.write_text(json.dumps(g));submitted=[];losses=[0]
  def output(args,**kw):
   if args[:3]==['scontrol','show','job']:return 'UserId=aclsylqidf(123) JobState=RUNNING NodeList='+('a14r2n20'if args[3]=='910407'else c.NODE)
   if args[:3]==['scontrol','show','hostnames']:return args[3]+'\n'
   assert args[0]=='sacct'
   if not submitted:return''
   if scenario=='observation_lost_then_complete'and losses[0]<2:losses[0]+=1;raise c.subprocess.TimeoutExpired(args,1)
   return '910374.999|'+c.LABEL+'|'+('FAILED|1:0'if scenario=='GPU_failed'else'COMPLETED|0:0')+'|'+c.NODE+'\n'
  def run(args,**kw):
   assert args[0]=='srun'and'--gres=none'in args,'Recording must not hide scancel/retry'
   return types.SimpleNamespace(returncode=0,stdout=json.dumps(dict(node=c.NODE,kfd_pids=[],HTTP43882_bindable=True,actual_step='910374.998'))+'\n')
  def popen(args,**kw):
   assert not submitted and args[0]=='srun'and'--jobid=910374'in args and'--gres=dcu:8'in args and'-c32'in args and'00:30:00'in args
   export=next(x for x in args if x.startswith('--export=')).split('=',1)[1].split(',');e=dict(x.split('=',1)for x in export[1:])
   assert e['WAVE_INSIDE_R2323']=='Deep-HC-teacher-forced-logits-r397/controller-v1/GPU_inside_r397.sh'and e['DEEP_SOURCE_MANIFEST_SHA']==c.OWNER_SHA
   assert all(kw['env'][k]==v for k,v in e.items()),'Actual Popen env must equal explicit export'
   assert float(e['DEEP_R277_GPUSTART_UNIX'])>0;submitted.append(args)
   report=dict(completed=True,passed=scenario!='driver_passed_false',actual_step='910374.999',legacy_tag='CPUrecord',normal_Native_shutdown=True,actual_forwards=368,actual_forward_attempts=368,actual_resets=4,actual_reset_attempts=4,HTTP_requests=0,HC_clear_rc=0,HC_after_clear=0,observer_disarm_rc=[0,0],cleanup_errors=[],owner_exit_observation=dict(all8_unique_threads_exited=True),CPU_FAKE_REPORT_NOT_GPU_QUALIFICATION=True,sequences=[dict(HC_mode=mode,completed=True,reset_ok=True,prefill_calls=9,decode_calls=83,decisions=[dict(output_index=i,finite=True)for i in range(84)])for mode in [0,1,1,0]])
   (root/'results/Deep-HC-teacher-forced-logits-r397-910374.999.json').write_text(json.dumps(report))
   return types.SimpleNamespace(pid=393784,poll=lambda:1 if scenario=='GPU_failed'else 0,wait=lambda timeout:1 if scenario=='GPU_failed'else 0)
  with mock.patch.object(c,'R',root),mock.patch.object(c,'W',root/'watch.json'),mock.patch.object(c,'verify_sources',side_effect=lambda actual:c.validate_scope(actual)),mock.patch.object(c.subprocess,'check_output',side_effect=output),mock.patch.object(c.subprocess,'run',side_effect=run),mock.patch.object(c.subprocess,'Popen',side_effect=popen),mock.patch.object(c.time,'sleep',return_value=None),mock.patch.object(c.socket,'gethostname',return_value='a14r2n20'),mock.patch.dict(c.os.environ,{'SLURM_JOB_ID':'910407','SLURM_STEP_ID':'999','WAVE_INSIDE_R2323':'wrong-old-value','DEEP_SOURCE_MANIFEST_SHA':'wrong-old-value','DEEP_R277_GPUSTART_UNIX':'0'}),mock.patch.object(sys,'argv',['controller_397.py','--execution-go',str(go)]):
   c.main();watch=json.loads(c.W.read_bytes());assert watch['completed']and watch['post_empty']and watch['actual_GPU']=='910374.999'and len(submitted)==1
   assert watch['Model_qualified']is False and watch['Diagnostic_qualified']==(scenario in ['complete','observation_lost_then_complete'])
  checks.append(dict(case='actual_controller_main_'+scenario,passed=True,one_recorded_GPU_submit=True,zero_real_submit=True,actual_export_and_env_equal=True,no_observation_replay=True))
print(json.dumps(dict(passed=True,checks=checks,actual_controller_main=True,remote_source_verification_mocked_only=True,actual_CPU792_binding_preserved_in_source=True,Slurm=False,GPU=False,Native_dlopen=False,new_compile=False)))
