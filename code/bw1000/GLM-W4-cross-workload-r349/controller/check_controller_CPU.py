"""Actual349 controller with recording Slurm/processes and real temporary files."""
import pathlib,sys,json,tempfile,types,copy,hashlib,time,traceback
from unittest.mock import patch
D=pathlib.Path(__file__).parent
if not(D/'controller_r349.py').is_file():D=D/'controller'
sys.path.insert(0,str(D))
import controller_r349 as C,artifact_visibility_r349 as A
OWNER=D.parent/'owner'
if not OWNER.is_dir():OWNER=C.S
result=dict(completed=False,passed=False,CPU_only=True,GPU=0,Native_Context=0,real_Slurm=False,cases=[])
def case(name,**data):result['cases'].append(dict(name=name,passed=True,**data))
def go():
 g=json.loads((D/'execution-GO.template.json').read_bytes());g.update(GPU_authorized=True,publication_commit='a'*40,controller_manifest_SHA256='b'*64)
 C.validate_scope(g);return g
try:
 g=go()
 for field,value in [('max_HTTP',6),('max_Native',19),('GPU_max_s',10801),('Native_max_s',10201),('CPU_keeper_max_s',12001),('GPU_parent',910371),('no_retry_reload',False)]:
  bad=copy.deepcopy(g);bad[field]=value
  try:C.validate_scope(bad)
  except(AssertionError,PermissionError):pass
  else:raise AssertionError('invalid scope accepted: '+field)
 case('exact_18_and_original_resource_bounds')
 for kind in ['missing','partial','stale','forever','final_bad_quality']:
  with tempfile.TemporaryDirectory()as td:
   p=pathlib.Path(td)/'report.json';clock=[0];sleeps=[]
   if kind=='partial':p.write_text('{')
   if kind=='stale':p.write_text('{"completed":false}')
   if kind=='final_bad_quality':p.write_text('{"completed":true,"passed":false}')
   def sleep(seconds):
    sleeps.append(seconds);clock[0]+=seconds
    if kind in ['missing','partial','stale']and clock[0]>=2:p.write_text('{"completed":true,"passed":true}')
   with patch.object(A.time,'monotonic',lambda:clock[0]),patch.object(A.time,'sleep',sleep):
    try:q=A.terminal_report(p)
    except TimeoutError:assert kind=='forever' and clock[0]==90
    else:assert q['completed'] and (q['passed']if kind!='final_bad_quality'else not q['passed'])
   if kind=='final_bad_quality':assert not sleeps
   case('bounded_visibility_'+kind,fake_wait_s=clock[0])
 for mode in ['normal','srun_nonzero','duplicate_empty_watch','duplicate_partial_watch','existing_label','bad_completed_owner']:
  with tempfile.TemporaryDirectory()as td:
   r=pathlib.Path(td);(r/'results').mkdir();(r/'logs').mkdir();(r/'control').mkdir();w=r/'results/watch.json';gp=r/'GO.json';gp.write_text(json.dumps(g));actual='910376.2345';starts=[];probes=[];stagefiles=[]
   if mode=='duplicate_empty_watch':w.write_bytes(b'')
   if mode=='duplicate_partial_watch':w.write_bytes(b'{')
   fixtures=json.loads((OWNER/'FIXTURE_CARD.json').read_bytes())['fixtures']
   rows=[]
   for fi,f in enumerate(fixtures):
    for ai,window in enumerate([2,4,2,4,4,2]):rows.append(dict(fixture_id=f['id'],fixture_index=fi,arm_index=ai,global_index=fi*6+ai,pair_index=fi*6+ai,prefill_control_window=window,paired_warmup=ai<2,prompt_tokens=f['current_prepared_prompt_tokens'],requested_output_budget=4096,generated_natural_EOS=True,reason='eos_token',actual_prompt_token_ids_SHA256=f['current_prepared_prompt_IDs_SHA256'],generated_token_ids_SHA256='c'*64,generated_tokens=1,original_reset_all32_completed=True))
   for stage in range(4):
    (r/'logs'/(C.LABEL+'-'+actual+'-'+str(stage)+'.log')).write_text('Using network IBext_v8\nGLM_REGULAR_SERVICE_R204_NORMAL '+str(stage)+'\n')
    for rank in range(8):
     q=dict(completed=True,worker_shutdown_completed=True,post_close_Torch_liveness=True,normal_reset_close=True,native_service_loop_completed=True,normal_shutdown=True,library_sha256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',latest_protocol_binding_r290={'source_manifest_SHA256':C.OWNER_SHA},actual_producer=actual,service_requests=rows,full_requests_completed=18,planned_Native=18,attempted_Native=18,completed_Native=18,normal_HTTP_shutdown=True,cleanup={'forced_cleanup':False,'errors':[]})
     if mode=='bad_completed_owner'and(stage,rank)==(0,0):q['normal_shutdown']=False
     p=r/'results'/('GLM-latest-full-owner-r290-'+C.LABEL+'-stage'+str(stage)+'-rank'+str(rank)+'.json');p.write_text(json.dumps(q));stagefiles.append(p)
   (r/'results'/('GLM-latest-full-owner-r290-supervisor-'+C.LABEL+'.json')).write_text(json.dumps(dict(completed=True,passed=True,planned_HTTP=18,planned_Native=18,attempted_HTTP=18,completed_HTTP=18,actual_HTTP=18,actual_Native=18,actual_GPU_handle=actual)))
   def output(argv,**kwargs):
    if argv[:3]==['scontrol','show','hostnames']:return '\n'.join(C.NODES)+'\n'
    assert argv[0]=='sacct'
    active=bool(starts)or mode=='existing_label'
    return '|'.join([actual,C.LABEL,'COMPLETED','0:0',','.join(C.NODES)])+'\n'if active else ''
   def run(argv,**kwargs):
    assert argv[0]=='srun' and '--gres=none'in argv and '-N4'in argv and '-n4'in argv;probes.append(argv)
    data=[dict(node=n,actual_step='910376.'+str(2344+len(probes)),CPU_only=True,GPU_runtime_initialized=False,KFD_empty=True,port43843_bindable=True)for n in C.NODES]
    return types.SimpleNamespace(returncode=0,stdout='\n'.join(json.dumps(x)for x in data))
   class Process:
    pid=7349002
    def __init__(self,argv,**kwargs):
     assert kwargs['start_new_session'];assert '--gres=dcu:8'in argv and '-N4'in argv and '-n4'in argv and '-c64'in argv and '03:00:00'in argv
     assert any('runtime_v4/owner/GPU_inside_r290.sh'in x for x in argv)
     assert 'GLM_PP_SIZE_R3603=4'in argv and 'GLM_CACHE_CAP_R3627=32768'in argv
     starts.append(argv)
    def poll(self):return 7 if mode=='srun_nonzero'else 0
   parents=[]
   patches=[patch.object(C,'R',r),patch.object(C,'W',w),patch.object(C,'S',OWNER),patch.object(C,'verify_sources',lambda x:C.validate_scope(x)),patch.object(C,'require_baseline_terminal',lambda *a:None),patch.object(C,'parent',lambda parent,node,env:parents.append((parent,node))),patch.object(C.socket,'gethostname',lambda:'a14r2n20'),patch.dict(C.os.environ,{'SLURM_JOB_ID':'910407','SLURM_STEP_ID':'999'}),patch.object(C.subprocess,'check_output',output),patch.object(C.subprocess,'run',run),patch.object(C.subprocess,'Popen',Process),patch.object(sys,'argv',['controller_r349.py','--execution-go',str(gp)])]
   for p in patches:p.start()
   try:
    try:C.main()
    except AssertionError:
     assert mode in ['duplicate_empty_watch','duplicate_partial_watch','existing_label','bad_completed_owner']
    else:assert mode not in ['duplicate_empty_watch','duplicate_partial_watch','existing_label','bad_completed_owner']
   finally:
    for p in reversed(patches):p.stop()
   if mode.startswith('duplicate')or mode=='existing_label':assert not starts
   else:
    assert len(starts)==1 and len(probes)==2
    receipt=json.loads(w.read_bytes());assert receipt['post_empty'] and receipt['terminal'][0]==actual
    assert receipt['scoped_Model_pair_qualified']==(mode=='normal')
    if mode=='normal':assert receipt['actual_HTTP']==receipt['actual_Native']==18 and len(receipt['owner_results'])==32
   case('actual_controller_recording_'+mode,recorded_GPU_srun_count=len(starts),real_GPU=False)
 result['passed']=True
except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
finally:result['completed']=True;print(json.dumps(result))
raise SystemExit(0 if result['passed']else 1)
