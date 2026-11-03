"""Only changed9741 window handoff/6plan and actual finalsourceCLI. No Model."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path('/refactor');D=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(D/'owner'))
from workload_r349 import load_plan,body_and_gold
from glm_prefill_execute_r236 import execute_plan
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 card,plan=load_plan(D/'owner/FIXTURE_CARD.json');assert len(card['fixtures'])==1 and [a['window']for a in plan]==[2,4,2,4,4,2]and[a['warmup']for a in plan]==[True,True,False,False,False,False]
 f=card['fixtures'][0];raw,body,gold=body_and_gold(f);p=Path(f['current_prepared_prompt_IDs']);assert sha(p)==f['current_prepared_private_IDs_file_SHA256'];ids=json.loads(p.read_bytes())['prompt_token_ids'];assert len(ids)==9741 and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==f['current_prepared_prompt_IDs_SHA256']
 chunks=[(start,ids[start:start+512])for start in range(0,len(ids),512)];assert len(chunks)==20 and len(chunks[-1][1])==13
 rows=[]
 for stage in range(2):
  for rank in range(8):
   for arm in plan:
    calls=[];votes=[];samples=[];discards=[];state=[0]
    def forward(past,values,final):assert past==state[0]and list(values)==ids[past:past+len(values)];calls.append([past,len(values),final]);state[0]+=len(values)
    def vote():votes.append(state[0]);return False
    def sample():samples.append(state[0]);return 17
    value=execute_plan(chunks,0,prefill=True,window=arm['window'],forward=forward,vote=vote,sample=sample,discard=lambda:discards.append(1))
    assert value==(9741,17,False)and len(calls)==20 and samples==[9741]and discards==[1]
    expected=[0]+[min(x*512,9741)for x in range(arm['window'],20+arm['window'],arm['window'])];assert votes==expected,(arm,votes,expected)
    rows.append(dict(stage=stage,rank=rank,arm=arm['global_index'],window=arm['window'],prefill_calls=20,final_past=9741,sample_count=1,WORLD_vote_pasts=votes,Native_executed=False))
 # Required changed failure boundaries: cancellation only after a full W4group,
 # and first actual forward exception discards without sampling/reset/replay.
 events=[];position=[0]
 def consume(p,v,final):position[0]+=len(v);events.append(['forward',p,len(v)])
 def cancel():events.append(['vote',position[0]]);return position[0]>=2048
 v=execute_plan(chunks,0,prefill=True,window=4,forward=consume,vote=cancel,sample=lambda:events.append('unexpected_sample'),discard=lambda:events.append('discard'))
 assert v==(2048,None,True)and events[-1]=='discard'and sum(isinstance(e,list)and e[0]=='forward'for e in events)==4
 failed=[]
 def throws(*args):failed.append('forward');raise RuntimeError('recorded Nativefailure at changedwindow leaf')
 try:execute_plan(chunks,0,prefill=True,window=4,forward=throws,vote=lambda:False,sample=lambda:failed.append('sample'),discard=lambda:failed.append('discard'))
 except RuntimeError:pass
 else:raise AssertionError('Forwardfailure swallowed')
 assert failed==['forward','discard']
 # Invoke the exact final CLI in a separate isolated interpreter. Source-only
 # mode reads allnewsource/real952/realpriorCPU820+850 receipts and executes
 # actualshell→productionpurelayout, then returns before Slurm/intent/Popen.
 mp=D/'GPU_MANIFEST.template.json';argv=[sys.executable,'-I','-B',str(D/'launch_once.py'),'--publication-commit',os.environ['GLM_R414_PUBLICATION'],'--manifest-file',mp.name,'--manifest-sha',sha(mp),'--source-preflight-only']
 proc=subprocess.run(argv,text=True,capture_output=True,timeout=100);assert proc.returncode==0,(proc.stdout,proc.stderr);cli=json.loads(proc.stdout);assert cli['source_preflight_passed']and cli['Slurm']==cli['GPU']==0 and not cli['Native_dlopen']and not cli['intent_created']
 return dict(completed=True,passed=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),owner_manifest_SHA256=sha(D/'owner/SOURCE_MANIFEST.json'),request_SHA256=f['request_SHA256'],prompt_IDs_SHA256=f['current_prepared_prompt_IDs_SHA256'],prompt_tokens=9741,ordinary_budget=4096,windows=[a['window']for a in plan],warmups=2,formal_indices=[2,3,4,5],actual_existing_window_consumer_SHA256=sha(D/'owner/glm_prefill_execute_r236.py'),recorded16owner_window_handoffs=rows,cancel_after_complete_W4group=True,exception_discard_without_replay=True,actual_final_CLI_argv=argv,actual_final_CLI=cli,GPU=0,Model=0,Native_dlopen=False,checkpoint_IO=False,compile=False,CPU804_820_850_repeated=False,scope='Changedproducer6plan→unchangedactualwindowconsumer oncomplete9741 officialIDs with recordedforward/control leaves;actualisolatedsourceCLI/bytes/parent952/receipts/Bashpure. No Native/math/weight/device qualification.')
if __name__=='__main__':
 out=R/'candidates/GLM-NoV-16card-W4-pair-r414-CPU';out.mkdir(exist_ok=False);p=out/'CPU-result.json';start=time.monotonic()
 try:q=run()
 except BaseException as e:
  import traceback;q=dict(completed=True,passed=False,error=repr(e),traceback=traceback.format_exc(),GPU=0,Native_dlopen=False,Model=0)
 q['elapsed_s']=time.monotonic()-start;p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(path=str(p),SHA256=sha(p),passed=q['passed'])));raise SystemExit(0 if q['passed']else 1)
