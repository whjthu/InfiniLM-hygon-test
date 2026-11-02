"""Actual controller main; Slurm/Popen/terminal-collector leaves recorded only."""
from pathlib import Path
import importlib.util,json,os,socket,subprocess,sys,tempfile,types
from unittest.mock import patch
D=Path(__file__).resolve().parent
def run():
 spec=importlib.util.spec_from_file_location('r404_actual_controller',D/'controller/controller.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
 import collect as collector
 outcomes=[]
 for fail_collect in [False,True]:
  with tempfile.TemporaryDirectory(prefix='r404-controller-')as temp:
   r=Path(temp);[(r/n).mkdir()for n in ['results','logs','control']];source=r/'incoming';source.mkdir();starts=[];probes=[];queries=[0]
   m=dict(GPU_GO=True,Model_GO=True,source_manifest_SHA256='c'*64,owner_manifest_SHA256='d'*64,CPU_receipt_SHA256='e'*64,Native_SHA256='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b')
   actual='910376.999991';label=C.LABEL+'-'+C.RUN
   class Process:
    pid=404001
    def __init__(self,argv,**kwargs):
     starts.append((argv,kwargs));self.returncode=None
     for stage in range(2):(r/'logs'/(label+'-'+actual+'-'+str(stage)+'.log')).write_text('GLM_REGULAR_SERVICE_R204_NORMAL '+str(stage)+'\n')
    def poll(self):return self.returncode
    def wait(self,timeout=None):self.returncode=0;return 0
   def output(argv,**kwargs):
    if argv[:3]==['scontrol','show','job']:return 'UserId=aclsylqidf(154459) JobState=RUNNING NodeList=a02r4n[12-13]'
    if argv[:3]==['scontrol','show','hostnames']:return 'a02r4n12\na02r4n13\n'
    assert argv[0]=='sacct'
    if not starts:return ''
    queries[0]+=1
    if queries[0]==1:raise subprocess.TimeoutExpired(argv,20)
    return '|'.join([actual,label,'COMPLETED','0:0','a02r4n[12-13]'])+'\n'
   def leaf_run(argv,**kwargs):
    assert argv[0]=='srun'and'--gres=none'in argv;probes.append(argv)
    rows=[dict(node=n,actual_step='910376.999990',CPU_only=True,GPU_initialized=False,KFD_available=True,kfd_pids=[],matching_component_processes=[],expected_PIDs_still_exist=[],ports_bindable={'43843':True})for n in C.NODES]
    return types.SimpleNamespace(returncode=0,stdout='\n'.join(json.dumps(x)for x in rows),stderr='')
   def terminal_leaf(*args):
    assert args[:3]==(r,C.RUN,actual)
    if fail_collect:raise RuntimeError('recording terminal metadata/quality refusal')
    return dict(normal16owners=True,expected_node_PIDs={n:[]for n in C.NODES},semantic_gold_and_EOS=True,recording_collector_only=True)
   with patch.object(C,'R',r),patch.object(C,'S',source),patch.object(C,'verify',lambda *a:m),patch.object(C.subprocess,'Popen',Process),patch.object(C.subprocess,'check_output',output),patch.object(C.subprocess,'run',leaf_run),patch.object(C.time,'sleep',lambda *a:None),patch.object(C.socket,'gethostname',lambda:'a14r2n20'),patch.object(collector,'collect',terminal_leaf),patch.dict(os.environ,{'SLURM_JOB_ID':'910407','SLURM_STEP_ID':'CPU-recording'}),patch.object(sys,'argv',['controller.py','--publication-commit','a'*40,'--manifest-sha','b'*64]):
    rc=C.main()
   watch=json.loads((r/'results'/(label+'-watch.json')).read_bytes())
   assert rc==(1 if fail_collect else 0)and watch['completed']and watch['passed']is(not fail_collect)
   assert len(starts)==1 and queries[0]>=2 and not watch['forced_cleanup']and watch['actual_GPU']==actual
   assert len(watch['transient_observation_errors'])==1
   argv,kwargs=starts[0];assert'-c64'in argv and'--mem=224000M'in argv and argv[argv.index('-t')+1]=='01:30:00'
   assert kwargs['env']['GLM_PP_SIZE_R3603']=='2'and kwargs['env']['GLM_RUN_R5040']==C.RUN
   assert float(kwargs['env']['GLM_R290_GPUSTART_UNIX'])==watch['original_GPU_start_unix']and watch['GPU_deadline_unix']==watch['original_GPU_start_unix']+5400
   outcomes.append(dict(name='collector_failure'if fail_collect else'transient_account_recovery_normal',passed=True,actual_controller_main=True,recording_only=True,Popen_count=1,retries=0,unchanged_epoch=True,actual_handle_preserved=True,failure_post=('post_after_failure'in watch),new_Slurm=False,GPU=False,Model=False))
 return dict(passed=True,cases=outcomes,actual_Popen_argv_and_main=True,Slurm_and_collector_recording=True,GPU=0,Model=0)
