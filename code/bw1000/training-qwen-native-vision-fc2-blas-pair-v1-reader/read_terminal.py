import datetime,hashlib,json,math,re,subprocess,traceback
from pathlib import Path
label='training-qwen-native-vision-fc2-blas-pair-v1'
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/label
request='30ea043c56824993a7d59d654ab1a6e8'
def load(name):
 p=R/name
 with p.open('rb') as stream: raw=stream.read((2<<20)+1)
 if not 0<len(raw)<=2<<20: raise ValueError('EVIDENCE_SIZE')
 return json.loads(raw),dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
def check(v,c):
 if not v: raise ValueError(c)
def main():
 t,pin=load('outcome.json')
 intent,ip=load('intent.json')
 check(intent['request_id']==request and t['epoch']==intent['epoch'],'REQUEST')
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 cols='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'
 a=subprocess.run(['sacct','-j','1103606','-n','-P','--format='+cols],capture_output=True,text=True,timeout=8,check=True)
 steps=[r[:7] for l in a.stdout.splitlines() if (r:=l.split('|'))[1] in (label,label+'-pre',label+'-post')]
 out=dict(UTC=now,request=request,actual_GPU_step='1103606.282',steps=steps,accounting_columns=cols.split(','),pins={'intent':ip,'outcome':pin},keeper_passed=t.get('keeper_passed'),rank_records={},successful_ranks=0,formal_training=False)
 if not t['keeper_passed']:
  out['error_class']=(t['error'] or {}).get('type')
  out['clean_post']=bool(t['after'] and t['after']['kfd_holder_count']==0 and not t['after']['owned_processes_remaining'])
  for name in ('native.log',label+'.log'):
   p=R/name
   if p.is_file():
    with p.open('rb') as f:
     f.seek(max(0,p.stat().st_size-(64<<10)));tail=f.read(64<<10).decode(errors='replace')
    out.setdefault('bounded_failures',{})[name]=[l.strip()[:155] for l in tail.splitlines() if re.search(r'^(?:\[rank\d+\]:\s*)?(?:ValueError|AssertionError|TypeError|RuntimeError|AttributeError|KeyError):',l.strip())][-12:]
  return out
 check(len(steps)==3 and {r[0] for r in steps}=={'1103606.281','1103606.282','1103606.283'} and all(r[2:4]==['COMPLETED','0:0'] and r[5]=='f13r1n10' for r in steps),'ACTUAL_COMPLETED_ACCOUNTING')
 check('gres/dcu=8' in next(r for r in steps if r[0]=='1103606.282')[6].split(','),'EIGHT_CARD_TRES')
 a=t['after'];b=t['before']
 check(a['boot_id']==b['boot_id'] and a['pci']==b['pci'] and len(a['pci'])==8 and a['kfd_holder_count']==0 and not a['owned_processes_remaining'] and a['master_port_available'],'CLEAN_OWNED_RESOURCES')
 assigned,x=load('assigned.json');out['pins']['assigned']=x
 check(assigned['step']=='1103606.282' and len(set(assigned['PCI_by_index']))==8,'ACTUAL_DEVICE_ASSIGNMENT')
 native,x=load('native-outcome.json');out['pins']['native_outcome']=x
 check(native['normal_native_exit'] and native['rank_record_counts']=={'repeat':8} and native['cleanup']['owned_processes_remaining']==0 and not native['cleanup']['termination_needed'],'NATIVE_TERMINAL')
 out['clean_post']=True
 for rank in range(8):
  f=f'repeat-rank-{rank:02d}.json';d,pin=load(f);out['pins'][f]=pin
  check(d['rank']==rank and d['matched_original_trainable_values']==504 and d['pinned_rank_rng_restored_for_every_forward'] and d['original_recorder_validated_and_detached'] and d['retained_grad_enabled'] and d['backward_calls']==d['optimizer_calls']==0,'PER_RANK_REPLAY_BINDING')
  rows=d['records']
  order=['cublas','cublaslt','cublaslt','cublas']
  check([v['fc2_preference'] for v in rows]==order,'ABBA')
  check(all(math.isfinite(v['loss']) and v['entry_rng']==d['original_rng_sha'] and v['math_sdpa_calls']>0 and v['vision_calls']==rows[0]['vision_calls'] for v in rows),'ORIGINAL_REPLAY_STATE')
  def view(a,b):
   def first(xs,ys):
    return next((t['boundary'] for t,s in zip(xs,ys) if t!=s),None)
   check(list(a['boundaries'])==list(b['boundaries']) and [k['boundary'] for k in a['detailed']]==[k['boundary'] for k in b['detailed']],'OBSERVATION_IDENTITY')
   return dict(logits_equal=a['logits']==b['logits'],loss_equal=a['loss']==b['loss'],abs_loss_diff=abs(a['loss']-b['loss']),
               first_detailed=first(a['detailed'],b['detailed']),
               first_decoder_boundary=next((key for key in a['boundaries'] if a['boundaries'][key]!=b['boundaries'][key]),None))
  for run,name in zip(rows,order):
   preference={'cublas':'_BlasBackend.Cublas','cublaslt':'_BlasBackend.Cublaslt'}[name]
   c=run['fc2_calls']
   check(27<=len(c)<=108 and len(c)%27==0 and all(call['layer']==i%27 and call['dtype']=='torch.bfloat16' and call['actual_preference']==preference for i,call in enumerate(c)),'FULL_VISION_FC2_SCOPE')
  out['rank_records'][str(rank)]=dict(input_sha=d['original_input_sha'],calls_per_occurrence=len(rows[0]['fc2_calls']),
     rows_per_occurrence=sorted({v['rows'] for v in rows[0]['fc2_calls']}),
     cublas=view(rows[0],rows[3]),cublaslt=view(rows[1],rows[2]),cross_first=view(rows[0],rows[1]))
 out['successful_ranks']=8
 out['actual_forwards']=32
 out['actual_backward']=0
 out['actual_updates']=0
 out['actual_kernel_dispatch_proven']=False
 return out
try:
 print(json.dumps(main(),allow_nan=False))
except FileNotFoundError:
 print(json.dumps({'outcome_present':(R/'outcome.json').is_file(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
except Exception as e:
 print(json.dumps(dict(read_error=type(e).__name__,line=traceback.extract_tb(e.__traceback__)[-1].lineno,detail=str(e) if isinstance(e,(ValueError,KeyError)) else None)))
