"""Actual TP8 prefix-state qualification. No MTP head or serving/TPS claim."""
import pathlib,json,hashlib,os,re,time,ctypes as C
R=pathlib.Path('/refactor');D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
runtime_path=pathlib.Path(os.environ['DEEP_PREFIX_RUNTIME_R129']);g=json.loads(runtime_path.read_bytes());assert g['explicit_native_GO']is True
assert g['parent']=='910374'and g['node']=='a02r4n05'and os.uname().nodename==g['node']and os.environ['SLURM_JOB_ID']==g['parent'];assert re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
assert sha(D/'manifest-r136.json')==g['source_manifest_SHA256'];m=json.loads((D/'manifest-r136.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
cp=pathlib.Path(g['CPU_receipt']);assert sha(cp)==g['CPU_receipt_SHA256']=='4a151a66a5149ad9b423b87c2a74e03e40bcaa3a1094bdb6eddad117b4fe9bf0';cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['GPU_initialized']and not cpu['dlopen'];assert cpu['source_manifest_SHA256']==g['native_source_manifest_SHA256']=='1e01674262efd3350cf05dca360188dfa3bb1beca03af3b53cecc0b86526e95d';assert sha(cpu['library'])==cpu['library_SHA256']==g['DSO_SHA256']=='f100e038c6e883e0210380724d7492810cbad1c20384ea4cd162dca4ca2326b2';assert {'modern_deep_prefix_inspect_r129','modern_deep_prefix_fault_r129'}<=set(cpu['required_exports'])
for path,proof in cpu['actual_DTK_bindings'].items():assert sha(path)==proof['SHA256'],path
import numpy as np
import torch
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09';assert sha(model/'model.safetensors.index.json')=='7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a'
cfg=json.loads((model/'config.json').read_bytes());assert cfg['num_hidden_layers']==43;mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];output=R/'results'/('Deep-prefix-native-r136-'+tag+'.json');assert not output.exists()
report={'completed':False,'normal_close':False,'actual_step':tag,'CPU_receipt_SHA256':sha(cp),'library_SHA256':cpu['library_SHA256'],'source_manifest_SHA256':sha(D/'manifest-r136.json'),'physical_rows':2,'capacity':256,'TP':8,'PP':1,'actual_owners':8,'Model_gain':'UNKNOWN','MTP_head':False,'raw_activations_exported':False,'cases':[],'controls':[],'hard_failures':[],'numerical_quality':'UNRESOLVED_REVIEW_CONTROLS','native_source_manifest_SHA256':g['native_source_manifest_SHA256'],'scope':'Native fixedcohort prefix state/validity/continuation qualification; constructed valid token inputs. Not QA/online/throughput qualification.'}
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
  a=f32(a);b=f32(b);assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all()
  rows=[]
  for aa,bb in zip(a,b):
   delta=aa-bb;den=max(1e-20,float(np.sqrt(np.mean(bb*bb))));ka=np.argsort(aa)[-8:][::-1];kb=np.argsort(bb)[-8:][::-1]
   pa=np.exp(aa-float(np.max(aa)));pa/=np.sum(pa);pb=np.exp(bb-float(np.max(bb)));pb/=np.sum(pb);mean=(pa+pb)*.5
   js=float(.5*np.sum(pa*np.log(np.maximum(pa,1e-30)/np.maximum(mean,1e-30)))+.5*np.sum(pb*np.log(np.maximum(pb,1e-30)/np.maximum(mean,1e-30))))
   rows.append({'NRMSE':float(np.sqrt(np.mean(delta*delta)))/den,'max_abs':float(np.max(np.abs(delta))),'cosine':float(np.dot(aa,bb))/max(1e-20,float(np.linalg.norm(aa)*np.linalg.norm(bb))),'argmax_candidate':int(ka[0]),'argmax_reference':int(kb[0]),'argmax_agreement':bool(ka[0]==kb[0]),'top8_candidate':[int(v)for v in ka],'top8_reference':[int(v)for v in kb],'top8_overlap':len(set(ka)&set(kb))/8,'softmax_JS_nats':js,'reference_top1_probability':float(pb[kb[0]]),'candidate_probability_of_reference_top1':float(pa[kb[0]])})
  delta=a-b;relative=float(np.sqrt(np.mean(delta*delta)))/max(1e-20,float(np.sqrt(np.mean(b*b))))
  return {'relative_RMS':relative,'rows':rows,'finite':True,'NRMSE_is_diagnostic_only':True,'exact_tokens_required':False,'model_quality':'UNQUALIFIED','interpretation':'Neither a numerical threshold nor token equality is a task-quality acceptance gate. State/causality investigation and natural QA remain required.'}
 import traceback
 current=None;fatal_recovery=False
 def checkpoint(phase,**values):
  current.update(phase=phase,**values);save()
 def case(kind,meta,fn,control=False):
  global current,fatal_recovery
  current={'kind':kind,**meta,'hard_state_passed':False,'phase':'starting'};report['controls'if control else'cases'].append(current);save()
  if fatal_recovery:current.update(phase='skipped_unrecoverable_prior_state',skipped=True);save();return
  try:fn();current.update(hard_state_passed=True,phase='complete');save()
  except Exception as error:
   current.update(error=repr(error),traceback=traceback.format_exc()[-5000:],phase='hard_failure');report['hard_failures'].append({'kind':kind,**meta,'error':repr(error)});save()
   try:reset();recovered=inspect([0,0]);current['recovered_by_whole_reset']=recovered;save()
   except Exception as recovery:current['recovery_error']=repr(recovery);fatal_recovery=True;save()
 def ordinary_sequence(p,counts,pair,replacement):
  reset();forward(prompt(p),[0,0]);forward([row[:c]for row,c in zip(pair,counts)],[p,p]);expected=[p+c for c in counts];state=inspect(expected);logits,ids=forward(replacement,expected);return state,logits,ids
 pair=[[tokens[1],tokens[2]],[tokens[3],tokens[4]]];replacement=[[tokens[5]],[tokens[6]]]
 # Same accepted inputs and exact shape under ordinary repeat establish a
 # reproducibility control. Full accept2/2 then compares identical T4 shapes
 # against the transactional route, separately from partial T3-vs-T4 tests.
 for p in [2,3,127]:
  def control(p=p):
   counts=[2,2];expected=[p+2,p+2];baseline,reference,reference_ids=ordinary_sequence(p,counts,pair,replacement);checkpoint('ordinary_reference',baseline=baseline)
   repeat_state,repeated,repeat_ids=ordinary_sequence(p,counts,pair,replacement);checkpoint('ordinary_repeat',ordinary_repeat=compare(repeated,reference),ordinary_repeat_ids=[int(v)for v in repeat_ids],reference_ids=[int(v)for v in reference_ids])
   reset();forward(prompt(p),[0,0]);begin(p);verify_s=verify(p,pair);unpublished=inspect([p,p]);commit_s=commit(counts);committed=inspect(expected);actual,actual_ids=forward(replacement,expected);checkpoint('matched_T4_full_accept',unpublished=unpublished,committed=committed,full_accept_same_shape=compare(actual,reference),candidate_ids=[int(v)for v in actual_ids],verify2_wall_s=verify_s,seal_publish_wall_s=commit_s)
  case('ordinary_repeat_and_matched_T4_full_accept',{'position':p,'accepted_inputs':[2,2]},control,True)
 # Change a rejected future input at the same packed T4 shape. This is a
 # causality diagnostic, not a proof that any observed change is BF16 drift:
 # routed-expert packing can still change with that future token.
 def future_control():
  p=2;counts=[1,2];expected=[3,4];outputs=[]
  for rejected in [tokens[2],tokens[7]]:
   altered=[[tokens[1],rejected],[tokens[3],tokens[4]]];reset();forward(prompt(p),[0,0]);begin(p);verify(p,altered);commit(counts);state=inspect(expected);out,ids=forward(replacement,expected);outputs.append(out);checkpoint('future_variant_'+str(len(outputs)),**{'variant_'+str(len(outputs)):{'state':state,'candidate_ids':[int(v)for v in ids]}})
  checkpoint('rejected_future_comparison',rejected_future=compare(outputs[1],outputs[0]),shape_equal=True,accepted_inputs_equal=True,quality='UNQUALIFIED')
 case('same_T4_rejected_future_causality_diagnostic',{'position':2,'accepted_inputs':[1,2]},future_control,True)
 for p,counts in [(p,c)for p in [2,3,126,127,128]for c in [[1,2],[2,1]]]:
  def partial(p=p,counts=counts):
   expected=[p+c for c in counts];baseline,reference,reference_ids=ordinary_sequence(p,counts,pair,replacement);checkpoint('ordinary_reference',baseline=baseline,reference_ids=[int(v)for v in reference_ids],reference_packed_T=sum(counts))
   reset();forward(prompt(p),[0,0]);begin(p);verify_s=verify(p,pair);unpublished=inspect([p,p]);checkpoint('unpublished_state',unpublished=unpublished,candidate_verify_T=4)
   wrong=arr(counts,np.uint64);assert commitfn(ctx,ticket['instance']+1,ticket['transaction'],ticket['generation'],2,ptr(wrong))!=0;assert commitfn(ctx,ticket['instance'],ticket['transaction'],ticket['generation']-1,2,ptr(wrong))!=0
   commit_s=commit(counts);committed=inspect(expected);checkpoint('committed_state',committed=committed,foreign_and_stale_tickets_rejected=True)
   actual,actual_ids=forward(replacement,expected);checkpoint('continuation_diagnostic',continuation=compare(actual,reference),candidate_ids=[int(v)for v in actual_ids],verify2_wall_s=verify_s,seal_publish_wall_s=commit_s)
  case('partial_accept',{'position':p,'accepted_inputs':counts},partial)
 for p in [3,127]:
  def cancel(p=p):
   reset();forward(prompt(p),[0,0]);reference,reference_ids=forward(replacement,[p,p]);checkpoint('cancel_reference',reference_ids=[int(v)for v in reference_ids]);reset();forward(prompt(p),[0,0]);begin(p);verify(p,[[tokens[1],eos],[tokens[3],tokens[4]]]);ck(cancelfn(ctx,ticket['instance'],ticket['transaction'],ticket['generation']));state=inspect([p,p]);checkpoint('cancel_state',state=state);actual,actual_ids=forward(replacement,[p,p]);checkpoint('cancel_continuation',continuation=compare(actual,reference),candidate_ids=[int(v)for v in actual_ids])
  case('cancel_rejected_EOS_then_replacement',{'position':p},cancel)
 for name,counts in [('accepted_EOS',[2,1]),('cap_after_first_input',[1,1])]:
  def stop(counts=counts):
   reset();forward(prompt(3),[0,0]);begin(3);verify(3,[[tokens[1],eos],[tokens[3],tokens[4]]]);commit(counts);state=inspect([3+c for c in counts]);checkpoint('stop_state',state=state);commit(counts,False);checkpoint('duplicate_rejected',stale_duplicate_commit_rejected=True,frontend_policy_simulated=True)
  case(name,{'accepted_inputs':counts},stop)
 for phase in [1,2]:
  def fault(phase=phase):
   reset();forward(prompt(127),[0,0]);begin(127);ck(faultfn(ctx,3,phase));verify(127,pair);commit([1,2],False);state=inspect([127,127],True,mixed=None if phase==1 else[[127,127],[128,129]]);status=json.loads(statusfn(ctx));assert status['poisoned'];counts=state['transaction_counts'];assert counts[1]==0 if phase==1 else 0<counts[1]<8;checkpoint('fault_state',state=state,cohort_status=status)
   reset();reset_state=inspect([0,0]);forward(prompt(3),[0,0]);forward(replacement,[3,3]);checkpoint('whole_reset_continuation',reset_state=reset_state,valid_continuation_after_whole_reset=True)
  case('preseal_fault'if phase==1 else'deterministic_partial_publication_fault',{'fault_phase':phase},fault)
 assert len(report['cases'])==16 and len(report['controls'])==4
 hard=all(c.get('hard_state_passed')for c in report['cases']);controls=all(c.get('hard_state_passed')for c in report['controls'])
 report.update(completed=True,all_state_cases_passed=hard,diagnostic_controls_complete=controls,numerical_quality='UNRESOLVED_REVIEW_CONTROLS',model_quality='UNQUALIFIED',MTP_performance='UNMEASURED');save()
 if not(hard and controls):raise RuntimeError('hard state/finite/recovery failure; retained all16 case statuses')
except BaseException as e:report['error']=repr(e);report['traceback']=traceback.format_exc()[-5000:]if 'traceback'in globals()else None;save();raise
finally:
 if ctx is not None:code=destroy(ctx);report['normal_close']=code==0;report['destroy_code']=code
 report['elapsed_s']=time.perf_counter()-started;save()
print('DEEP_PREFIX_R129_NORMAL',json.dumps({'completed':report['completed'],'normal_close':report['normal_close']}),flush=True)
