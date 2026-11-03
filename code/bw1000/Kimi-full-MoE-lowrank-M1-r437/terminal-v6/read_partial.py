"""Original769 failed-run audit. Stdlib only; no GPU/job/codec execution."""
from pathlib import Path
import argparse,base64,hashlib,json,math,re,struct
sha=lambda b:hashlib.sha256(b).hexdigest()
def view(shape,strides,new):
 # Source replay of actual TensorImpl::view merge/split, limited to this nonempty positive case.
 assert math.prod(shape)==math.prod(new)
 merged=[shape[0]];steps=[strides[0]]
 for n,s in zip(shape[1:],strides[1:]):
  if s*n==steps[-1]:merged[-1]*=n;steps[-1]=s
  else:merged.append(n);steps.append(s)
 at=0;step=steps[0];left=merged[0];out=[]
 for n in new:
  while n>left:at+=1;step=steps[at];left=merged[at]
  assert left%n==0;out.append(step*(left//n));left//=n
 return out

def audit(observation,rawfile,viewfile,linearfile,nonefile):
 o=json.loads(Path(observation).read_bytes());r=json.loads(Path(rawfile).read_bytes());v=json.loads(Path(viewfile).read_bytes());b=json.loads(Path(linearfile).read_bytes());n=json.loads(Path(nonefile).read_bytes())
 wraw=o['watch']['rawtext'].encode();assert sha(wraw)==o['watch']['SHA256'];w=json.loads(wraw)
 assert w['completed']and not w['passed']and w['actual_step']=='910371.769'and w['forced_cleanup']and w['owned_failure_cleanup']['closed']
 assert '910371.769|Kimi-r437-full-MoE-lowrank-v4|FAILED|6:0|'in o['accounting']['stdout']
 assert '910407.899|Kimi-r437-full-MoE-lowrank-v4-keeper|FAILED|1:0|'in o['accounting']['stdout']
 assert '910371.770|Kimi-r437-full-MoE-lowrank-v4-failure-post|COMPLETED|0:0|'in o['accounting']['stdout']
 assert len(w['failure_post'])==2 and all(x['passed']and not x['KFD_pids']and not x['matching_GPU_driver_PIDs']and not x['expected_PIDs_still_exist']and x['passive_master_port40917_bindable']for x in w['failure_post'])
 data={};rows=[]
 for path,item in r['raw_files'].items():
  raw=item['rawtext'].encode()if path.endswith('.json')else base64.b64decode(item['base64']);assert len(raw)==item['bytes']and sha(raw)==item['SHA256'];data[path]=raw
 for path,raw in data.items():
  if not path.endswith('.json'):continue
  m=re.search(r'node([01])-raw/rank([0-7])[.]json$',path);assert m;node,rank=map(int,m.groups());q=json.loads(raw)
  assert not q['completed']and q['node_rank']==node and q['local_rank']==rank and q['global_tp_rank']==node*8+rank and q['TP']==16 and q['PP']==1
  assert q['error']=='437 only original exact C1 BF16 contiguous lowrank projection'
  assert q['Runtime_created_mask']==q['entry_mask']and len(q['entry_mask'])>=32 and q['owner_mask']==q['pinned_mask']==q['entry_mask'][rank*4:rank*4+4]
  assert len(q['arms'])==1;a=q['arms'][0];assert a['arm']==a['mode']==0 and a['warm']and a['nonfinite']==0 and a['input_and_tail_canary_unchanged']and a['down_up_selected_delta']==[0,0]and a['global_barrier_sum']==16 and a['raw_output_elements']==7168
  output=data[path[:-5]+'-arm0.bf16'];assert len(output)==14336
  values=[struct.unpack('<f',struct.pack('<I',z[0]<<16))[0]for z in struct.iter_unpack('<H',output)];assert all(map(math.isfinite,values));rms=math.sqrt(sum(x*x for x in values)/len(values));assert rms>0
  rows.append(dict(node=node,local_rank=rank,global_rank=node*8+rank,owner_mask=q['owner_mask'],entry_mask=q['entry_mask'],raw_output_SHA256=sha(output),finite=True,RMS=rms,warm_OFF_full_MoE_ms=a['full_MoE_forward_and_completion_ms'],warm_OFF_checked_ms=a['checked_call_including_output_readback_save_release_ms'],input_canary_passed=True))
 assert sorted(x['global_rank']for x in rows)==list(range(16))and len(data)==32
 for node in[0,1]:assert len({c for x in rows if x['node']==node for c in x['owner_mask']})==32
 sources={}
 for source in[v,b,n]:
  for path,item in source['source'].items():assert sha(item['rawtext'].encode())==item['SHA256'];sources[path]=item
 aliases=[x for p,x in sources.items()if p.endswith('/tensor/view.cc')];assert len(aliases)==2 and len({x['SHA256']for x in aliases})==1
 tensor=next(x for p,x in sources.items()if p.endswith('/tensor/tensor.cc'));provider=next(x for p,x in sources.items()if p.endswith('/none_quantization.cpp'))
 assert 'calculate_contiguous_strides(shape)'in tensor['rawtext']and 'auto input_contiguous = input->contiguous(runtime)'in provider['rawtext']
 derived=view([1,1,7168],[14336,7168,1],[1,7168]);assert derived==[14336,1]
 return dict(audit_passed=True,run_passed=False,state='FAIL_PRESERVED_NO_COMPLETE_ON_PAIR',actual_GPU='910371.769 FAILED6/77s',actual_keeper='910407.899 FAILED1/84s',post='910371.770 COMPLETED0;bothKFD_PIDempty_portbindable',original_epoch=w['original_start'],GPU_manifest_SHA256=w['GPU_source_manifest_SHA256'],completed_OFF_warm_records=16,completed_ON_records=0,formal_records=0,actual_projection_ON_hits='UNKNOWN_not_persisted_before_guard',baseline_OFF_outputs=rows,normal16=False,failure_cleanup_closed=True,Model=False,performance='UNKNOWN_no_formal_pair',math_failure='NOT_ESTABLISHED_guard_not_numeric',observed_descriptor_gaps=['which-projection and each guard predicate','input exact shape,strides,offset,dtype,device,contiguity before guard','actual resident weight descriptor','ON/output descriptor absent'],source_derived_not_observed=dict(storage_shape=[1,2,7168],storage_strides=[14336,7168,1],narrow_shape=[1,1,7168],narrow_strides=[14336,7168,1],flatten_shape=[1,7168],flatten_strides=derived,offset=0,logical_input_bytes=14336,physical_single_row_bytes=14336,strict_is_contiguous=False,unused_singleton_row_stride=14336,address_formula='for row0, element j at data+2*j for 0<=j<7168; rowstride contributes0',original_provider='NoneQuantization validates then contiguous(runtime) before dense_projection',claim='Deterministic source replay only; actual firstON descriptors were not saved, do not fabricate them'),source_SHA256={p:x['SHA256']for p,x in sources.items()},next='STOP device; source descriptor/address contract closure first; no blanket strided/3D acceptance',GPU_GO=False,Model_GO=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--raw',required=True);p.add_argument('--view',required=True);p.add_argument('--linear',required=True);p.add_argument('--none',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.observation,a.raw,a.view,a.linear,a.none);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps({k:q[k]for k in ['audit_passed','run_passed','actual_GPU','completed_OFF_warm_records','completed_ON_records','formal_records','performance']}))
