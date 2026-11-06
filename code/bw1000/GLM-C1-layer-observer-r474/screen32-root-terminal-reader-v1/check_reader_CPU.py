"""Boundary recordings for the reader; never reads real906/900 artifacts."""
import copy,math
from read_original_terminal import accounting,successful,bucket,clocks,equal_optional,frame_report,schedule,phases,request,JH,PROFILE_NAMES,SELECTED,screen_summary,collector_state,GPU,LABEL,audit_post,NODES
checks=0

def reject(f):
 global checks
 try:f()
 except(AssertionError,ValueError,KeyError,StopIteration):checks+=1;return
 raise AssertionError('invalid evidence accepted')

cols=['JobIDRaw','State','ExitCode','ElapsedRaw']
a=accounting(dict(accounting_columns=cols,accounting=['910371.906|COMPLETED|0:0|4000','910407.1121|FAILED|1:0|4001']))
assert successful(a['910371.906'])and not successful(a['910407.1121'])
a=accounting(dict(accounting_columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw'],accounting=['910371.906|original|COMPLETED|0:0|4000']))
assert successful(a['910371.906'])
reject(lambda:accounting(dict(accounting_columns=cols,accounting=['910371.906|RUNNING|0:0|1','910371.906|COMPLETED|0:0|2'])))
w=dict(passed=False,GPU=GPU,terminal_visibility_complete=False,terminal_accounting_observations=[None,[GPU,LABEL,'RUNNING','0:0','4000']],GPU_accounting=[GPU,LABEL,'RUNNING','0:0','4000'])
cs=collector_state(w);assert cs['original_observed_states']==['MISSING','RUNNING']and not cs['original_passed']
x=copy.deepcopy(w);x.update(passed=True,srun_rc=0);reject(lambda:collector_state(x))
assert len(schedule())==33 and len(phases())==37 and sum(not a['diagnostic']for a in schedule())==32
assert schedule()[0]['case_index']==schedule()[1]['case_index']==0 and schedule()[0]['diagnostic']and not schedule()[1]['diagnostic']
bucket({},0,10,[],False)
reject(lambda:bucket({'count':0},0,10,[],False))
# A true final1token prefill at fragment5 is prefill, but original selection omits it.
fr=[dict(tokens=512,offset=i*512)for i in range(4)]+[dict(tokens=1,offset=2048)]
b=dict(count=5,sum_s=[5.]*6,min_s=[1.]*6,max_s=[1.]*6,selected=[dict(index=i+1,tokens=512,past=i*512,seconds=[1.]*6)for i in range(4)])
bucket(b,5,2049,fr,True)
x=copy.deepcopy(b);x['selected'].append(dict(index=5,tokens=1,past=2048,seconds=[1.]*6));reject(lambda:bucket(x,5,2049,fr,True))
c=clocks([2],[100],[2],[],50);assert c['visible']==0 and c['rate']is None and c['TTFT_s']is None
c=clocks([5],[100],[],[],50);assert c['visible']==1 and c['rate']is None and c['TTFT_s']==50/1e9
c=clocks([5,6,2],[100,200,300],[2],[],50);assert c['visible']==2 and c['rate']==1e7 and c['TTFT_s']==50/1e9
reject(lambda:clocks([5,6],[100,90],[],[],50))
reject(lambda:clocks([5],[49],[],[],50))
reject(lambda:equal_optional(None,1.))

# Original header stores legacy local device0 while outer typed report is globalTP rank8.
marks=[dict(name='CAPI.owner.begin',owner_host_s=100.,gpu_from_first_s=0.)]
for l in range(78):
 for tag in ['layer.%d.%s.begin'%(l,'full'if l%4==0 else'shared'),'layer.%d.end'%l]:
  j=len(marks);marks.append(dict(name=tag,owner_host_s=100+j*.001,gpu_from_first_s=j*.0008))
f=dict(tokens=1,stage=0,rank=0,row_past_count=[[0,10,1]],sequence=3,event_count=len(marks),full_detail_layer=0,shared_detail_layer=1,marks=marks)
r=dict(global_tp_rank=8,enabled=True,frame_cap=3,event_cap_per_frame=512,events_allocated=512,pending_next_decode=False,total_frames_used=1,samples=[f],coverage=[0,0,0,1],required=[0,0,0,3],missing_is_UNKNOWN=True,sequence=3)
v=frame_report(r,8,10,2,2);assert not v['coverage_complete']and v['coverage']=='UNKNOWN_SHORT_NATURAL_DIAGNOSTIC'and len(v['frames'][0]['layer_enclosures'])==78
x=copy.deepcopy(r);x['global_tp_rank']=0;reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x['samples'][0]['rank']=8;reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x['samples'][0]['row_past_count']=[[0,9,1]];reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x['samples'][0]['event_count']=251;reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x['samples'][0]['marks'][2]['name']='missing-layer0-end';reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x['samples'][0]['marks'][2]['gpu_from_first_s']=-1.;reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x['total_frames_used']=4;reject(lambda:frame_report(x,8,10,2,2))
x=copy.deepcopy(r);x.update(sequence=2,total_frames_used=0,samples=[],coverage=[0,0,0,0]);assert frame_report(x,8,10,2,1)['coverage']=='UNKNOWN_SHORT_NATURAL_DIAGNOSTIC'

# A full4096 nonEOS cap is retained rather than rejected to improve rate/quality.
prompt=[7]*513;fr=[dict(offset=o,tokens=len(prompt[o:o+512]),IDs_SHA256=JH(prompt[o:o+512]))for o in range(0,len(prompt),512)]
def make_bucket(count,prefill):
 if not count:return{}
 selected=[]
 for i in range(1,count+1):
  tokens=fr[i-1]['tokens']if prefill else 1
  if tokens>1 or i in SELECTED:selected.append(dict(index=i,tokens=tokens,past=fr[i-1]['offset']if prefill else len(prompt)+i-1,seconds=[.001]*6))
 return dict(count=count,sum_s=[.001*count]*6,min_s=[.001]*6,max_s=[.001]*6,selected=selected[:24])
ids=[5]*4096;ticks=[1000000000+(i+1)*1000000 for i in range(4096)];t=clocks(ids,ticks,[2],[],1000000000)
frozen=dict(ordinal=1,case_id='CPU-cap-fixture',suite='gsm8k',prompt_tokens=len(prompt),prompt_IDs_SHA256=JH(prompt),body_SHA256='fixedCPU')
plan=dict(frozen,prompt_token_ids=prompt,max_new_tokens=4096,prefill_window=4096,body={'temperature':0.0})
r=dict(frozen,completed=True,reset_before=True,reset_after=True,generated_token_ids=ids,sampled_ns=ticks,prefill_fragment_plan=fr,planned_max_new_tokens=4096,runtime_prefill_chunk=512,original_request_prefill_window=4096,diagnostic=False,observer_diagnostic=False,screen_index=0,arm='screen',pair=-1,warmup=False,owner_cores=4,owner_CPUmask=[0,1,2,3],pipeline_profile_names=PROFILE_NAMES,generated_token_ids_SHA256=JH(ids),natural_EOS=False,cap_hit=True,Native_forward_begin=0,Native_forward_end=4097,past_after=4608,prefill_CAPI_s=[.001]*2,decode_CAPI_s=[.001]*4095,pipeline_prefill=make_bucket(2,True),pipeline_decode=make_bucket(4095,False),request_enter_ns=1000000000,visible_nonEOS_nonspecial=4096,backend_decode_tok_s=t['rate'],backend_sampled_TTFT_s=t['TTFT_s'],whole_backend_s=5.,observer_mark_calls_before=3,observer_mark_calls_after=3)
row,frames,fc,cc=request(r,plan,frozen,dict(diagnostic=False,screen_index=0),1,0,list(range(64)),{'eos_token_ids':[2],'tokenizer':{'special_ids':[]}},0)
assert row['cap_hit']and not row['natural_EOS']and fc==4097 and cc==4096 and frames is None
ss=screen_summary([dict(diagnostic=False,correct=False,parse_failure=True,natural_EOS=False,cap_hit=True,backend_decode_tok_s=15.),dict(diagnostic=False,correct=True,parse_failure=False,natural_EOS=True,cap_hit=False,backend_decode_tok_s=16.)])
assert ss==dict(planned=32,validated=2,correct=1,parse_failure=1,natural_EOS=1,cap_hit=1,missing_or_unvalidated=30,strict_gt15=1)

# A genuine post may be clean even when Model evidence failed. Audit it independently.
import json
expected={};owners=[];workers=[];nodes=[];before=[]
for ni,node in enumerate(NODES):
 lead=1000+ni*100;children=[dict(PID=lead+i+1,startticks=str(i+2),global_rank=ni*8+i)for i in range(8)]
 expected[node]=[dict(PID=lead,startticks='1',socket_inodes=[])]+[dict(PID=c['PID'],startticks=c['startticks'],socket_inodes=[])for c in children]
 workers.append(dict(PID=lead,startticks='1',owners=children))
 owners.extend(dict(node=node,PID=c['PID'],startticks=c['startticks'],owned_socket_inodes=[],passed=False)for c in children)
 pci=['CPUfixture-%d-%d'%(ni,i)for i in range(8)];before.append(dict(node=node,physical_PCI=pci));nodes.append(dict(node=node,actual_step='910371.907',KFD_available=True,kfd_pids=[],matching_Model_processes=[],expected_owned_PIDs_still_live=[],expected_UIDsocket_listeners_still_live=[],ports_bindable={'43843':True,'43844':True},physical_PCI=pci))
w=dict(before={'nodes':before},failure_post=dict(raw=dict(rc=0,argv=['env','GLM470_EXPECTED_PIDS='+json.dumps(expected)]),nodes=nodes),passed=False)
a=accounting(dict(accounting_columns=cols,accounting=['910371.907|COMPLETED|0:0|1']))
assert audit_post(w,a,owners,workers,strict=False)=='910371.907'and w['passed']is False
x=copy.deepcopy(w);x['failure_post']['nodes'][0]['KFD_available']=False;reject(lambda:audit_post(x,a,owners,workers))
x=copy.deepcopy(w);x['failure_post']['nodes'][0]['ports_bindable'].pop('43844');reject(lambda:audit_post(x,a,owners,workers))
x=copy.deepcopy(w);x['failure_post']['nodes'][0]['expected_owned_PIDs_still_live']=[1001];reject(lambda:audit_post(x,a,owners,workers))
x=copy.deepcopy(owners);x[0]['startticks']='wrong';reject(lambda:audit_post(w,a,x,workers))
assert checks==18
print('PASS: 18 malformed-evidence rejections; independent post/Model failure, keeperFAIL,33/32/37schedule,tail1prefill,EOS/unknownrate/TTFT,shortframes/local-global rank, full4096cap and32denominator. CPU fixtures only; no original906 consumption.')
