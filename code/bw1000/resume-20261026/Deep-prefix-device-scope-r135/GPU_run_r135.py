"""Actual TP8 prefix-state qualification. No MTP head or serving/TPS claim."""
import pathlib,json,hashlib,os,re,time,ctypes as C
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
runtime_path=pathlib.Path(os.environ['DEEP_PREFIX_RUNTIME_R129']);g=json.loads(runtime_path.read_bytes());assert g['explicit_native_GO']is True
assert g['parent']=='910374'and g['node']=='a02r4n05'and os.uname().nodename==g['node']and os.environ['SLURM_JOB_ID']==g['parent'];assert re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
assert sha(D/'manifest-r135.json')==g['source_manifest_SHA256'];m=json.loads((D/'manifest-r135.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
cp=pathlib.Path(g['CPU_receipt']);assert sha(cp)==g['CPU_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU_initialized']and not cpu['dlopen'];assert cpu['source_manifest_SHA256']==g['source_manifest_SHA256'];assert sha(cpu['library'])==cpu['library_SHA256']==g['DSO_SHA256'];assert {'modern_deep_prefix_inspect_r129','modern_deep_prefix_fault_r129'}<=set(cpu['required_exports'])
for path,proof in cpu['actual_DTK_bindings'].items():assert sha(path)==proof['SHA256'],path
import numpy as np
import torch
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';assert sha(model/'model.safetensors.index.json')=='7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a'
cfg=json.loads((model/'config.json').read_bytes());assert cfg['num_hidden_layers']==43;mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];output=R/'results'/('Deep-prefix-native-r135-'+tag+'.json');assert not output.exists()
report={'completed':False,'normal_close':False,'actual_step':tag,'CPU_receipt_SHA256':sha(cp),'library_SHA256':cpu['library_SHA256'],'source_manifest_SHA256':sha(D/'manifest-r135.json'),'physical_rows':2,'capacity':256,'TP':8,'PP':1,'actual_owners':8,'Model_gain':'UNKNOWN','MTP_head':False,'raw_activations_exported':False,'cases':[],'scope':'Native fixedcohort prefix state/validity/continuation qualification; constructed valid token inputs. Not QA/online/throughput qualification.'}
def save():p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
lib=C.CDLL(cpu['library']);V=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64
def bind(n,args,restype=I):f=getattr(lib,n);f.argtypes=args;f.restype=restype;return f
err=bind('modern_native_engine_error',[],C.c_char_p);create=bind('modern_native_engine_create',[C.c_char_p,Z,Z],V);keysfn=bind('modern_native_engine_keys',[V],C.c_char_p);load=bind('modern_native_engine_load',[V,C.c_char_p]);finish=bind('modern_native_engine_finish',[V]);resetfn=bind('modern_native_engine_reset',[V]);destroy=bind('modern_native_engine_destroy',[V]);ordinary=bind('modern_native_engine_forward',[V,Z,Z]+[V]*8)
beginfn=bind('modern_deep_prefix_begin_r129',[V,U,Z,V,V]);ticketfn=bind('modern_deep_prefix_ticket_r129',[V],C.c_char_p);verifyfn=bind('modern_deep_prefix_verify_r129',[V,U,U,U,Z,V,V,V,V]);commitfn=bind('modern_deep_prefix_commit_r129',[V,U,U,U,Z,V]);cancelfn=bind('modern_deep_prefix_cancel_r129',[V,U,U,U]);inspectfn=bind('modern_deep_prefix_inspect_r129',[V,Z,V],C.c_char_p);faultfn=bind('modern_deep_prefix_fault_r129',[V,I,I]);statusfn=bind('modern_deep_prefix_status_r128',[V],C.c_char_p)
def ck(code):
 if code:raise RuntimeError(err().decode())
def arr(v,kind):return np.ascontiguousarray(v,dtype=kind)
def ptr(v):return V(v.ctypes.data)
def f32(v):return (v.astype(np.uint32)<<16).view(np.float32)
ctx=None;started=time.perf_counter();tx=0;ticket=None
try:
 # Explicit current safe policy5 is FP32 reductions, never old F64 modes2..4.
 ck(bind('modern_deep_precision_policy_r3443',[I])(5));ck(bind('modern_deep_layout_policy_r7052',[I])(0));ck(bind('deep_hc_canonical_policy_r7130',[I])(0));ck(bind('deep_moe_tail_set_r7211',[I])(0));ck(bind('modern_deep_c4_tile_policy_r5920',[I])(0))
 device_contract=json.loads(bind('modern_deep_prefix_device_contract_r135',[],C.c_char_p)());report['device_scope_contract']=device_contract;save();assert device_contract['passed']and device_contract['actual_owners']==8;ctx=create(json.dumps(cfg).encode(),2,256);assert ctx,err();keys=set(json.loads(keysfn(ctx)));assert len(keys)==67569 and keys<=set(mapping)
 grouped={}
 for key in keys:grouped.setdefault(mapping[key],[]).append(key)
 pending=[];pending_bytes=0;loaded=0
 def flush():
  global pending_bytes,loaded
  if not pending:return
  ds=[{'name':name,'pointer':value.data_ptr(),'shape':list(value.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.int8:2,torch.int64:3}[value.dtype]}for name,value in pending];ck(load(ctx,json.dumps(ds).encode()));loaded+=len(pending);pending.clear();pending_bytes=0
 for filename,names in sorted(grouped.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as f:
   for name in sorted(names):
    value=f.get_tensor(name).contiguous();pending.append((name,value));pending_bytes+=value.numel()*value.element_size()
    if len(pending)>=1024 or pending_bytes>=512*1024*1024:flush()
   flush()
 ck(finish(ctx));report['loading_seconds']=time.perf_counter()-started;report['loaded_keys']=loaded;save()
 tokenizer=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True);tokens=tokenizer.encode('One two three four five six seven eight nine ten. A short causal sequence for cache validation.',add_special_tokens=False);assert tokens and all(0<=v<cfg['vocab_size']for v in tokens);eos=tokenizer.eos_token_id;assert isinstance(eos,int)
 def reset():ck(resetfn(ctx))
 def forward(seqs,past):
  widths=[len(v)for v in seqs];offsets=[0]
  for width in widths:offsets.append(offsets[-1]+width)
  values=[arr(v,np.int32)for v in [[0,1],past,[p+w for p,w in zip(past,widths)],offsets]]+[arr([x for seq in seqs for x in seq],np.int64),arr([p+i for p,w in zip(past,widths)for i in range(w)],np.int64)];logits=np.empty((2,cfg['vocab_size']),dtype=np.uint16);ids=np.empty(2,dtype=np.int64);ck(ordinary(ctx,offsets[-1],2,*[ptr(v)for v in values],ptr(logits),ptr(ids)));assert np.isfinite(f32(logits)).all();return logits,ids
 def prompt(p):
  seq=[tokens[i%len(tokens)]for i in range(p)];return [seq,list(reversed(seq))]
 def inspect(expected,poison=False,mixed=None):
  rows=arr([0,1],np.uint64);raw=inspectfn(ctx,2,ptr(rows));assert raw,err();q=json.loads(raw);owners=q['actual_owners'];assert len(owners)==8;lengths=[];validity=[]
  for rank,a in enumerate(owners):
   assert a['scoped_device_binding']is True and 0<=a['previous_current_device']<8;assert a['actual_rank']==a['Runtime_device']==a['current_device']==rank and a['configured_rows']==2 and len(a['layers'])==43 and a['model_poisoned']==poison and a['free_bytes']>=4294967296
   want=a['model_lengths']if mixed is not None else expected
   if mixed is not None:assert want in mixed
   assert a['native_outer_lengths']==a['model_lengths']==want;lr=[];vr=[]
   for l,layer in enumerate(a['layers']):
    assert layer['layer']==l and layer['ratio']==int(cfg['compress_ratios'][l]);one=[];valid=[]
    for row,z in enumerate(layer['rows']):
     L=want[row];assert z['row']==row and z['model_length']==z['decoder_length']==z['attention_length']==L and z['ring']['ring_validity_matches_committed_length']and z['ring']['valid_ring_slots']==min(L,128);one.append(L)
     if layer['ratio']:
      c=z['compressor'];assert c['length']==L and c['validity_matches_committed_groups']and c['valid_compressed_slots']==L//layer['ratio']
     if layer['ratio']==4:
      i=z['indexer'];assert i['length']==i['compressor_length']==L and i['validity_matches_committed_groups']and i['valid_groups']==L//4
     valid.append(True)
    lr.append(one);vr.append(valid)
   lengths.append(lr);validity.append(vr)
  sampled=q.get('sampled_free_bytes_8_by_5',[]);assert all(v>=4294967296 for row in sampled for v in row if v)
  return {'metadata_SHA256':hashlib.sha256(raw).hexdigest(),'rank_layer_row_lengths':lengths,'all_43_validity_checks':all(all(all(x)for x in layer)for layer in validity),'transaction_counts':q.get('transaction_counts'),'cohort_poisoned':q.get('cohort_poisoned'),'sampled_free_bytes_8_by_5':sampled,'sampled_not_peak':True}
 def begin(p):
  global tx,ticket
  tx+=1;rows=arr([0,1],np.int32);past=arr([p,p],np.int32);ck(beginfn(ctx,tx,2,ptr(rows),ptr(past)));ticket=json.loads(ticketfn(ctx));assert ticket["transaction"]==tx
 def verify(p,pair):
  packed=arr([v for row in pair for v in row],np.int64);positions=arr([p,p+1,p,p+1],np.int64);logits=np.empty((4,cfg['vocab_size']),dtype=np.uint16);ids=np.empty(4,dtype=np.int64);t=time.perf_counter();ck(verifyfn(ctx,ticket["instance"],ticket["transaction"],ticket["generation"],2,ptr(packed),ptr(positions),ptr(logits),ptr(ids)));assert np.isfinite(f32(logits)).all();return time.perf_counter()-t
 def commit(counts,success=True):
  counts=arr(counts,np.uint64);t=time.perf_counter();code=commitfn(ctx,ticket["instance"],ticket["transaction"],ticket["generation"],2,ptr(counts));
  if success:ck(code)
  else:assert code!=0
  return time.perf_counter()-t
 def compare(a,b):
  a=f32(a);b=f32(b);delta=a-b;relative=float(np.sqrt(np.mean(delta*delta))/max(1e-20,float(np.sqrt(np.mean(b*b)))));assert relative<0.01,relative;return {'relative_RMS':relative,'tolerance':.01,'max_abs':float(np.max(np.abs(delta))),'argmax_agreement_diagnostic':float(np.mean(a.argmax(1)==b.argmax(1))),'exact_tokens_required':False}
 cases=[(p,c)for p in [2,3,126,127,128]for c in [[1,2],[2,1]]]
 for p,counts in cases:
  pair=[[tokens[1],tokens[2]],[tokens[3],tokens[4]]];replacement=[[tokens[5]],[tokens[6]]]
  reset();forward(prompt(p),[0,0]);forward([row[:c]for row,c in zip(pair,counts)],[p,p]);expected=[p+c for c in counts];baseline_state=inspect(expected);reference,_=forward(replacement,expected)
  reset();forward(prompt(p),[0,0]);begin(p);verify_s=verify(p,pair);unpublished=inspect([p,p]);wrong=arr(counts,np.uint64);assert commitfn(ctx,ticket["instance"]+1,ticket["transaction"],ticket["generation"],2,ptr(wrong))!=0;assert commitfn(ctx,ticket["instance"],ticket["transaction"],ticket["generation"]-1,2,ptr(wrong))!=0;commit_s=commit(counts);committed=inspect(expected);actual,_=forward(replacement,expected)
  report['cases'].append({'position':p,'accepted_inputs':counts,'baseline':baseline_state,'unpublished':unpublished,'committed':committed,'continuation':compare(actual,reference),'verify2_wall_s':verify_s,'seal_publish_wall_s':commit_s});save()
 for p in [3,127]:
  reset();forward(prompt(p),[0,0]);reference,_=forward([[tokens[5]],[tokens[6]]],[p,p]);reset();forward(prompt(p),[0,0]);begin(p);verify(p,[[tokens[1],eos],[tokens[3],tokens[4]]]);ck(cancelfn(ctx,ticket["instance"],ticket["transaction"],ticket["generation"]));state=inspect([p,p]);actual,_=forward([[tokens[5]],[tokens[6]]],[p,p]);report['cases'].append({'kind':'cancel_rejected_EOS_then_replacement','position':p,'state':state,'continuation':compare(actual,reference)});save()
 # Accepted EOS/cap are frontend stop choices: qualify exact accepted INPUT
 # state, then reset. Do not fabricate target-generated EOS or service behavior.
 for name,counts in [('accepted_EOS',[2,1]),('cap_after_first_input',[1,1])]:
  reset();forward(prompt(3),[0,0]);begin(3);verify(3,[[tokens[1],eos],[tokens[3],tokens[4]]]);commit(counts);state=inspect([3+c for c in counts]);commit(counts,False);report['cases'].append({'kind':name,'state':state,'stale_duplicate_commit_rejected':True,'frontend_policy_simulated':True});save()
 for phase in [1,2]:
  reset();forward(prompt(127),[0,0]);begin(127);ck(faultfn(ctx,3,phase));verify(127,[[tokens[1],tokens[2]],[tokens[3],tokens[4]]]);commit([1,2],False);state=inspect([127,127],True,mixed=None if phase==1 else[[127,127],[128,129]]);status=json.loads(statusfn(ctx));assert status['poisoned'];counts=state['transaction_counts'];assert counts[1]==0 if phase==1 else 0<counts[1]<8
  reset();reset_state=inspect([0,0]);forward(prompt(3),[0,0]);forward([[tokens[5]],[tokens[6]]],[3,3]);report['cases'].append({'kind':'preseal_fault'if phase==1 else'deterministic_partial_publication_fault','state':state,'reset_state':reset_state,'valid_continuation_after_whole_reset':True});save()
 report.update(completed=True,all_cases_passed=True,model_quality='UNQUALIFIED',MTP_performance='UNMEASURED');save()
except BaseException as e:report['error']=repr(e);save();raise
finally:
 if ctx is not None:code=destroy(ctx);report['normal_close']=code==0;report['destroy_code']=code
 report['elapsed_s']=time.perf_counter()-started;save()
print('DEEP_PREFIX_R129_NORMAL',json.dumps({'completed':report['completed'],'normal_close':report['normal_close']}),flush=True)
