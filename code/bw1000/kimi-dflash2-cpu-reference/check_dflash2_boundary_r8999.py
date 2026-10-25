import pathlib,json,hashlib,copy
import numpy as np
from dflash2_boundary_r8999 import *
r=pathlib.Path(__file__).resolve().parent;c=json.loads((r/'config.json').read_text());descriptor=check_config(c)
rejected=[]
for field,value in [('target_layer_ids',[7,23,51,67,83]),('target_layer_ids',[19,37,54,66,78,90]),('attention_mode','gqa'),('mask_token_id',163824)]:
 bad=copy.deepcopy(c);bad['dflash_config'][field]=value
 try:check_config(bad)
 except AssertionError:rejected.append([field,value])
 else:raise AssertionError('Old/incompatible draft accepted')
taps=[np.full((1,1,H),i,np.uint16)for i in range(5)];assert feature_packet(taps).shape==(1,1,35840)
for old,new in [('fc.weight','context_proj.weight'),('hidden_norm.weight','context_norm.weight'),('norm.weight','final_norm.weight')]:assert new in serving_to_training_state({old:object()},object())
assert len(existing_spec_CAPI_arrays(verify_plan(10,[11,12],255,1,65)))==8
plan=verify_plan(10,[11,12],255,1,65);assert plan['slot_mapping']==[511,512,513]and plan['block_table']==[1,2,3,4]
tiny=[np.ones((1,2,2),np.float32)*(i+1)for i in range(5)];assert np.isfinite(context_reference(tiny,np.ones((2,10),np.float32),np.ones(2,np.float32))).all()
conv=np.arange(12,dtype=np.float32).reshape(1,3,4);base=np.ones((2,4),np.float32);dyn=np.zeros((1,3,2,2),np.float32);expected=conv.copy();expected[:,1:]+=conv[:,:-1];assert np.array_equal(grouped_convolve_reference(conv,dyn,base,2),expected)
hidden=np.ones((1,2,2),np.float32);logits=np.asarray([[[0,1,2,3],[3,2,1,0]]],np.float32);pre=np.ones((4,1),np.float32);suc=np.asarray([[0],[2],[-2],[0]],np.float32);proj=np.ones((1,2),np.float32)
assert selector_greedy_reference(hidden,logits,[0],pre,suc,proj,4).tolist()==[[1,1]]
# Negative and every accepted-prefix length: original active state is never
# mutated by verify/replay, even though rejected suffix corrupts the shadow.
cases=[]
for accepted in range(8):
 anchor=10;proposals=list(range(11,18));target=proposals[:accepted]+[100]+[101]*(7-accepted);states={1:(7,()),65:(0,())};calls=[]
 def callback(plan):
  before=states[plan['mamba_read']];value,history=before
  for token in plan['tokens']:value=(value*31+token)%1000003;history=(*history,token)
  states[plan['mamba_write']]=(value,history);calls.append(plan)
  return target[:len(plan['tokens'])]
 result=execute_paired_target_round(callback,anchor,proposals,23,1,65,enabled=True)
 assert states[1]==(7,())and len(states[65][1])==accepted+1
 assert len(calls)==(1 if accepted==7 else 2)and result['next_past']==24+accepted
 cases.append({'accepted':accepted,'calls':len(calls),'active_unchanged':True,'committed_state_inputs':len(states[65][1])})
stop=greedy_commit(10,[11,12],[11,163586,19],23,1,65,eos={163586});assert stop['terminal']and stop['emitted']==[11,163586]and stop['replay_inputs']is None
try:execute_paired_target_round(lambda p:[],10,[],0,1,65)
except RuntimeError:defaultoff=True
else:raise AssertionError('defaultOFF bypass')
out={'CPU_only':True,'GPU':False,'weights_read':False,'passed':True,'config':descriptor,'incompatible_configs_rejected':rejected,'prefix_cases':cases,'defaultOFF':defaultoff,'conv_equation':True,'selector_equation':True,'strict_state_key_remap':True,'scope':'NumPy boundary/equation/control tests only, NOT trained MLA forward, actual checkpoint header, target KDA state math, final tokens or performance qualification','remaining':['Native current Math0/TP16 verifier+replay is not qualified by historical Math11 TP8 PP4','TorchSpec actual MLA+YaRN dependency closure not installed/executed here','New checkpoint header/weights not read; strict load and tokenizer mapping not exercised']}
(r/'CPU-reference-result.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out,indent=2))
