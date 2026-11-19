import collections,datetime,hashlib,json,math,re,subprocess,traceback
from pathlib import Path
label='training-qwen-native-decoder-six-ten-boundaries-v1'
request='69317ea2bd3b46cbafd0a0be9d989056'
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/label
def get(name):
 path=root/name
 with path.open('rb') as f: raw=f.read((2<<20)+1)
 if not 0<len(raw)<=2<<20:raise ValueError('RECEIPT_BOUND')
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def require(ok,reason):
 if not ok:raise ValueError(reason)
def main():
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
 result=subprocess.run(['sacct','-n','-P','-j','1103606','--format='+cols],check=True,timeout=8,capture_output=True,text=True)
 rows=[r[:7] for s in result.stdout.splitlines() if (r:=s.split('|'))[1] in (label,label+'-pre',label+'-post')]
 byname={r[1]:r for r in rows}
 data=dict(UTC=now,request=request,accounting_columns=cols.split(','),steps=rows,
           complete=False,accounting_duplicate=len(byname)!=len(rows),planned_ranks=8,planned_forwards_per_rank=8)
 if not (root/'outcome.json').is_file():return data
 intent,ip=get('intent.json');outcome,op=get('outcome.json')
 require(intent['request_id']==request and intent['epoch']==outcome['epoch'] and not data['accounting_duplicate'],'ORIGINAL_SCOPE')
 data.update(keeper_passed=outcome['keeper_passed'],pins={'intent':ip,'controller':op},
   cleanup=bool(outcome['before'] and outcome['after'] and outcome['before']['pci']==outcome['after']['pci'] and outcome['before']['boot_id']==outcome['after']['boot_id']
     and len(outcome['after']['pci'])==8 and outcome['after']['kfd_holder_count']==0
     and not outcome['after']['owned_processes_remaining'] and outcome['after']['master_port_available']))
 if not outcome['keeper_passed']:
  data['error_class']=(outcome['error'] or {}).get('type')
  for path in (root/'native.log',root/(label+'.log')):
   if path.is_file():
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-(64<<10)));tail=f.read(64<<10).decode(errors='replace')
    data.setdefault('scoped_failures',{})[path.name]=[line.strip()[:160] for line in tail.splitlines()
        if re.search(r'^(?:\[rank\d+\]:\s*)?(?:ValueError|AssertionError|TypeError|RuntimeError|AttributeError|KeyError|OutOfMemoryError):',line.strip())][-14:]
  return data
 require(len(rows)==3 and all(r[2:4]==['COMPLETED','0:0'] and r[5]=='f13r1n10' for r in rows)
         and 'gres/dcu=8' in byname[label][6].split(',') and data['cleanup'],'ORIGINAL_GPU_AND_CLEAN_POST')
 native,pin=get('native-outcome.json');data['pins']['native']=pin
 require(native['normal_native_exit'] and native['rank_record_counts']=={'repeat':8} and native['native_returncode']==0,'NATIVE_RESULTS')
 assigned,pin=get('assigned.json');data['pins']['assigned']=pin
 require(assigned['step']==byname[label][0] and len(set(assigned['PCI_by_index']))==8,'ACTUAL_ASSIGNED_DEVICES')
 printables={}
 for rank in range(8):
  name=f'repeat-rank-{rank:02d}.json';doc,pin=get(name);data['pins'][name]=pin
  require(doc['rank']==rank and doc['matched_original_trainable_values']==504
          and doc['pinned_rank_rng_restored_for_every_forward'] is True and doc['original_recorder_validated_and_detached'] is True
          and doc['backward_calls']==doc['optimizer_calls']==0,'RANK_ORIGINAL_TRAINING_INPUT')
  all_rows=doc['records']
  target=rank in (1,5)
  require(len(all_rows)==8 and len(doc['forward_anchor_comparisons'])==7 and doc['decoder_focus']==([6,10] if target else []),'TARGET_LAYERS')
  baseline=all_rows[0];bydetail=[v['boundary'] for v in baseline['detailed']]
  if target:
   require(len(bydetail)==len(set(bydetail)) and len(bydetail) in ((48,49) if rank==1 else (42,))
      and all(sum(x.startswith(f'decoder_{i:02d}.') for x in bydetail)==21 for i in (6,10)),'EXACT_TARGET_DETAIL_COVERAGE')
  else:require(not bydetail,'UNTARGETED_RANK_ACTIVATIONS')
  touched=collections.Counter();retained=[]
  pairs=[]
  for i,row in enumerate(all_rows):
   require(row['fc2_preference']=='cublaslt' and all(c['actual_preference']=='_BlasBackend.Cublaslt' for c in row['fc2_calls'])
           and row['entry_rng']==doc['original_rng_sha'] and row['vision_calls']==baseline['vision_calls']
           and row['math_sdpa_calls']==baseline['math_sdpa_calls'] and math.isfinite(row['loss']),
           'ACTUAL_FIXED_FORWARD_CONTEXT')
   require(list(row['boundaries'])==list(baseline['boundaries']) and [v['boundary'] for v in row['detailed']]==bydetail
           and 0<=row['retained_logical_tensor_bytes']<=536870912,'EXACT_BOUNDARIES')
   retained.append(row['retained_logical_tensor_bytes'])
   if i==0:continue
   first=next((a['boundary'] for a,b in zip(baseline['detailed'],row['detailed']) if a!=b),None)
   whole=next((name for name,v in baseline['boundaries'].items() if v!=row['boundaries'][name]),None)
   declaration=doc['forward_anchor_comparisons'][i-1]
   require(declaration['logits_equal']==(row['logits']==baseline['logits'])
           and declaration['loss_equal']==(row['loss']==baseline['loss'])
           and declaration['first_different_boundary']==whole
           and declaration['first_different_detailed_boundary']==first,'INDEPENDENT_PAIR')
   if first:touched[first]+=1
   pairs.append(dict(repeat=i+1,logits_equal=declaration['logits_equal'],loss_equal=declaration['loss_equal'],
        abs_loss_difference=abs(row['loss']-baseline['loss']),first_whole_boundary=whole,first_detailed_boundary=first))
  printables[str(rank)]=dict(logits_changed=sum(not v['logits_equal'] for v in pairs),loss_changed=sum(not v['loss_equal'] for v in pairs),
      observed_detail=len(bydetail),first_detail_histogram=dict(touched),max_retained_logical_tensor_bytes=max(retained),
      pairs=pairs if target else None)
 data.update(complete=True,actual_GPU_step=byname[label][0],rank_records=printables,actual_forwards=64,actual_updates=0,
    formal_training=False,actual_backend_kernel_proven=False,
    stable_ranks=sum(not r['logits_changed'] and not r['loss_changed'] for r in printables.values()))
 return data
try:
 print(json.dumps(main(),allow_nan=False))
except Exception as exc:
 print(json.dumps(dict(reader_error=type(exc).__name__,message=str(exc)[:180],line=traceback.extract_tb(exc.__traceback__)[-1].lineno)))
