import datetime,hashlib,json,math,re,subprocess,traceback
from pathlib import Path
label='training-qwen-native-text-base-blas-pair-v1'
request='ab5458146c6c4997ad83ffad2ee0d2a2'
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/label
SELECTED={1:['decoder_06.v_proj'],5:['decoder_10.q_proj','decoder_06.mlp_down_proj']}
SITE_NAMES={f'decoder_{i:02d}.{name}' for i in range(36) for name in
            ('q_proj','k_proj','v_proj','o_proj','mlp_gate_proj','mlp_up_proj','mlp_down_proj')}
ARMS=['cublas','cublaslt','cublaslt','cublas']
PREF_NAMES={'cublas':'_BlasBackend.Cublas','cublaslt':'_BlasBackend.Cublaslt'}
def get(name):
 with (root/name).open('rb') as f: raw=f.read((4<<20)+1)
 if not 0<len(raw)<=4<<20:raise ValueError('RECEIPT_BOUND')
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def require(v,reason):
 if not v:raise ValueError(reason)
def main():
 fmt='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
 a=subprocess.run(['sacct','-n','-P','-j','1103606','--format='+fmt],check=True,capture_output=True,text=True,timeout=8)
 rows=[d[:7] for s in a.stdout.splitlines() if (d:=s.split('|'))[1] in (label,label+'-pre',label+'-post')]
 data=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),request=request,steps=rows,accounting_columns=fmt.split(','),
           complete=False,planned_ranks=8,planned_forwards_per_rank=4)
 if not (root/'outcome.json').is_file():return data
 intent,pin=get('intent.json');controller,c=get('outcome.json')
 require(intent['request_id']==request and intent['epoch']==controller['epoch'],'ORIGINAL_REQUEST')
 data.update(keeper_passed=controller['keeper_passed'],pins=dict(intent=pin,controller=c))
 post=controller.get('after');before=controller.get('before')
 data['cleanup']=bool(post and before and post['pci']==before['pci'] and post['boot_id']==before['boot_id']
                     and post['kfd_holder_count']==0 and post['master_port_available'] and not post['owned_processes_remaining'])
 if not data['keeper_passed']:
  data['error_class']=(controller.get('error') or {}).get('type')
  for name in ('native.log',label+'.log'):
   p=root/name
   if p.is_file():
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-(64<<10)));tail=f.read(64<<10).decode(errors='replace')
    data.setdefault('small_signatures',{})[name]=[s.strip()[:160] for s in tail.splitlines()
      if re.search(r'^(?:\[rank\d+\]:\s*)?(?:ValueError|TypeError|AssertionError|RuntimeError|AttributeError|KeyError):',s.strip())][-10:]
  return data
 names={r[1]:r for r in rows}
 require(len(rows)==len(names)==3 and all(r[2:4]==['COMPLETED','0:0'] and r[5]=='f13r1n10' for r in rows)
         and 'gres/dcu=8' in names[label][6].split(',') and data['cleanup'],'TERMINAL_GPU_AND_POST')
 assigned,pin=get('assigned.json');data['pins']['assigned']=pin
 require(assigned['step']==names[label][0] and len(set(assigned['PCI_by_index']))==8,'EIGHT_ORIGINAL_CARDS')
 native,nativ_pin=get('native-outcome.json');data['pins']['native']=nativ_pin
 require(native['normal_native_exit'] and native['rank_record_counts']=={'repeat':8} and native['native_returncode']==0,'ACTUAL_NATIVE_TERMINAL')
 by_rank={}
 for rank in range(8):
  key=f'repeat-rank-{rank:02d}.json';doc,pin=get(key);data['pins'][key]=pin
  require(doc['rank']==rank and doc['matched_original_trainable_values']==504 and doc['pinned_rank_rng_restored_for_every_forward']
          and doc['backward_calls']==doc['optimizer_calls']==0,'ACTUAL_ORIGINAL_RANK')
  inventory=doc['base_inventory']
  require(set(inventory)==SITE_NAMES and all(x['base_weight_dtype']=='torch.bfloat16' and x['lora_weight_dtype']=='torch.float32'
          and len(x['weight_shape'])==2 for x in inventory.values()),'ORIGINAL_252_FROZEN_WEIGHT_INVENTORY')
  runs=doc['records']
  require([x['text_base_preference'] for x in runs]==ARMS,'ORDER')
  detail=[v['boundary'] for v in runs[0]['detailed']]
  target=rank in SELECTED
  require(len(detail)==len(set(detail)) and len(detail) in ((54,55) if rank==1 else ((54,) if rank==5 else (0,))),'TRACE_EVENT_COVERAGE')
  geometry=None
  for row in runs:
   arm=row['text_base_preference']
   require(row['fc2_preference']=='cublaslt' and all(c['actual_preference']=='_BlasBackend.Cublaslt' for c in row['fc2_calls'])
     and 27<=len(row['fc2_calls'])<=108 and len(row['fc2_calls'])%27==0
     and row['entry_rng']==doc['original_rng_sha'] and row['vision_calls']==runs[0]['vision_calls']
     and math.isfinite(row['loss']) and 0<=row['retained_logical_tensor_bytes']<=536870912,'ORIGINAL_FIXED_FORWARD')
   seq=row['text_base_calls']
   require(len(seq)==252 and {v['site'] for v in seq}==SITE_NAMES and all(v['preference']==PREF_NAMES[arm] for v in seq),'ACTUAL_252_BASE_CALLS')
   current=[{k:v for k,v in point.items() if k!='preference'} for point in seq]
   require(geometry is None or current==geometry,'UNCHANGED_ACTUAL_INPUT_GEOMETRY')
   geometry=current
   require([v['boundary'] for v in row['detailed']]==detail and list(row['boundaries'])==list(runs[0]['boundaries']),
          'EXACT_OBSERVED_ORDER')
  pairs={}
  for arm,left,right in (('cublas',0,3),('cublaslt',1,2),('cross_first',0,1)):
   u=runs[left];v=runs[right]
   first=next((x['boundary'] for x,y in zip(u['detailed'],v['detailed']) if x!=y),None)
   layer=next((k for k,val in u['boundaries'].items() if val!=v['boundaries'][k]),None)
   d1={x['boundary']:x['value'] for x in u['detailed']}
   d2={x['boundary']:x['value'] for x in v['detailed']}
   actual=dict(logits_equal=u['logits']==v['logits'],loss_equal=u['loss']==v['loss'],
               first_different_boundary=layer,first_different_detailed_boundary=first)
   declared=doc['base_preference_pair_comparisons'][arm]
   require(all(declared[k]==v for k,v in actual.items()),'INDEPENDENT_COMPARISON')
   branches={}
   for site in SELECTED.get(rank,[]):
    branch={k:dict(input_equal=d1[f'{site}.{k}.input']==d2[f'{site}.{k}.input'],
                   output_equal=d1[f'{site}.{k}.output']==d2[f'{site}.{k}.output'])
            for k in ('base','lora_A','lora_B')}
    branch['combined_output_equal']=d1[site]==d2[site]
    branches[site]=branch
   actual.update(loss_diff=abs(u['loss']-v['loss']),branches=branches)
   pairs[arm]=actual
  by_rank[str(rank)]=dict(pairs=pairs,peft_provider_SHA256=doc['peft_linear_source_SHA256'],
    inherited=doc['inherited_blas_preference'],logical_retained_bytes_max=max(r['retained_logical_tensor_bytes'] for r in runs),detail_events=len(detail))
 stable={arm:sum(v['pairs'][arm]['logits_equal'] and v['pairs'][arm]['loss_equal'] for v in by_rank.values())
         for arm in ('cublas','cublaslt','cross_first')}
 providers={v['peft_provider_SHA256'] for v in by_rank.values()};require(len(providers)==1,'PEFT_PROVIDER_IDENTICAL_ALL_RANKS')
 data.update(complete=True,actual_GPU_step=names[label][0],physical_cards=8,forwards=32,actual_base_linear_calls=8*4*252,
             actual_updates=0,actual_backward=0,rank_results=by_rank,stable_logits_loss_by_arm=stable,formal_training=False,actual_kernel_dispatch_proven=False)
 return data
try:
 print(json.dumps(main(),allow_nan=False))
except Exception as e:
 print(json.dumps(dict(reader_error=type(e).__name__,message=str(e)[:180],line=traceback.extract_tb(e.__traceback__)[-1].lineno)))
