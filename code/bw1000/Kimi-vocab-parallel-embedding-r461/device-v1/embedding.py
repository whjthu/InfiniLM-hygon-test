"""Exact four-query Native component entry, shared by device and libffi recording.

NumPy/source tensor leaves are CPU-only. No Native is loaded at import time.
"""
import ctypes as C
import hashlib,json,struct,time
KEY='language_model.model.embed_tokens.weight'
P=C.c_void_p;U16=C.POINTER(C.c_uint16);I64=C.POINTER(C.c_int64)
SIGNATURES={
 'modern_kimi_embedding_probe_load_r461':([P,C.c_char_p],C.c_int),
 'modern_kimi_embedding_probe_r461':([P,C.c_uint64,I64,C.c_size_t,U16,C.c_size_t],C.c_int),
 'modern_kimi_vocab_embedding_info_r461':([],C.c_char_p),
 'modern_kimi_engine_output_metrics_r2475':([P],C.c_char_p),
 'modern_kimi_engine_error_r2475':([],C.c_char_p)}
def queries():
 edges=[0]+[x for rank in range(1,32)for x in (5120*rank-1,5120*rank)]+[163839]
 assert len(edges)==64
 return [[0],[163839],edges,edges*8]
def ID_SHA(ids):return hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()
def declare(lib):
 for name,(args,result)in SIGNATURES.items():fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
 return lib
def read(lib,name,*args):
 raw=getattr(lib,name)(*args)
 if not raw:raise RuntimeError(name+': missing JSON')
 return json.loads(raw)
def check(lib,code):
 if code:
  raw=lib.modern_kimi_engine_error_r2475()
  raise RuntimeError(raw.decode()if raw else'Native failed without message')
def census(lib,node,completed,tokens,backings=None):
 value=read(lib,'modern_kimi_vocab_embedding_info_r461')
 assert value['TP']==32 and value['full_vocabulary']==163840 and value['hidden']==7168
 assert len(value['owners'])==8
 addresses=[]
 for d,owner in enumerate(value['owners']):
  rank=node*8+d
  assert owner['device']==d and owner['constructed']and not owner['destroyed']
  assert owner['tp_rank']==rank and owner['physical_shape']==[5120,7168]
  assert owner['parameter_bytes']==73400320 and owner['registered_weight_aliases_same_tensor']
  assert (owner['global_row_begin'],owner['global_row_end_exclusive'])==(rank*5120,(rank+1)*5120)
  assert owner['forward_calls']==owner['BF16_sum_collectives']==owner['completed_fences']==completed
  assert owner['tokens']==tokens and owner['postload_checked']==bool(completed)
  assert type(owner['backing_address'])is int and owner['backing_address']>0
  addresses.append(owner['backing_address'])
 if backings is not None:assert addresses==backings,'Actual parameter backing changed across load/lookup'
 return value,addresses
def metrics(np,output,reference):
 actual=(output.astype(np.uint32)<<16).view(np.float32)
 expected=(reference.astype(np.uint32)<<16).view(np.float32)
 assert bool(np.isfinite(expected).all())and bool(np.isfinite(actual).all()),'Nonfinite embedding reference/output'
 delta=actual-expected
 error=float(np.sqrt(np.sum(delta*delta,dtype=np.float32)))
 norm=float(np.sqrt(np.sum(expected*expected,dtype=np.float32)))
 scale=max(1.,float(np.max(np.abs(expected))))
 return dict(finite=True,NRMS=error/max(norm,1e-30),maxabs=float(np.max(np.abs(delta))),maxabs_over_max1_reference=float(np.max(np.abs(delta)))/scale,
  differing_BF16_words=int(np.count_nonzero(output!=reference)),total_BF16_words=int(output.size),reference_F32_norm=norm)
def run(lib,ctx,source,*,node,configuration,record,memory,store,clock=time.time,deadline,monotonic=time.perf_counter):
 """The whole component work entry; source owns its full CPU tensor until return.

source.reference(ids) returns actual checkpoint CPU BF16 rows on device runs.
Recording tests substitute only this source and Native leaves, never device qualification.
 """
 import numpy as np
 declare(lib)
 def remaining():
  assert clock()<deadline,'Original non-sliding component work deadline'
 def snapshot(label):
  values=memory();assert len(values)==8
  assert all(total==configuration['physical_VRAM_bytes']and free>=configuration['floor_bytes']for free,total in values),'Original 4GiB floor'
  record(label,values);return values
 remaining();before,backings=census(lib,node,0,0);record('embedding_before_load',before)
 descriptor=source.descriptor
 assert descriptor['name']==KEY and descriptor['dtype']==1 and descriptor['shape']==[163840,7168]and descriptor['pointer']>0
 assert source.metadata['CPU_tensor_bytes']==2348810240
 record('CPU_embedding_source',source.metadata)
 begin=monotonic();check(lib,lib.modern_kimi_embedding_probe_load_r461(ctx,json.dumps([descriptor],separators=(',',':')).encode()))
 record('embedding_native_load_inclusive',dict(seconds=monotonic()-begin,keys=1,CPU_source_bytes=2348810240,
  local_destination_bytes_per_rank=73400320,scope='actual engine.load_params/Weight.select_source/8owner H2D+original completion; not pure DMA'))
 snapshot('memory_after_embedding_load');after,_=census(lib,node,0,0,backings);record('embedding_after_load',after)
 # Two CPU-native pre-rejections. Neither publishes epoch1; its successful
 # reuse for the first real lookup and unchanged census are checked below.
 bad_output=np.full(8*7168,0x5aa5,dtype=np.uint16)
 for bad in [-1,163840]:
  remaining();ids=(C.c_int64*1)(bad)
  code=lib.modern_kimi_embedding_probe_r461(ctx,1,ids,1,bad_output.ctypes.data_as(U16),bad_output.size)
  assert code!=0 and bool((bad_output==0x5aa5).all())
  unchanged,_=census(lib,node,0,0,backings)
  record('invalid_global_ID_pre_rejected',dict(ID=bad,requested_epoch=1,returncode=code,census=unchanged,
   GPU_lookup_submitted=False,epoch_nonpublication_evidence='valid first query must still accept epoch1; counters remain0'))
 results=[];total_tokens=0
 fixed=queries();assert [len(ids)for ids in fixed]==configuration['lookup_widths']==[1,1,64,512]
 for index,ids_list in enumerate(fixed):
  remaining();started=monotonic();width=len(ids_list);epoch=index+1
  ids=(C.c_int64*width)(*ids_list);reference=source.reference(ids_list)
  assert reference.dtype==np.uint16 and reference.shape==(width,7168)
  reference_record=store('reference-q%d.bf16'%epoch,reference.tobytes(order='C'))
  record('lookup_before',dict(epoch=epoch,tokens=width,IDs_SHA256=ID_SHA(ids_list),reference=reference_record,memory=snapshot('memory_before_query%d'%epoch)))
  elements=8*width*7168;guard=64
  output=np.full(elements+2*guard,0x5aa5,dtype=np.uint16)
  pointer=C.cast(int(output.ctypes.data)+guard*2,U16)
  native_start=monotonic();check(lib,lib.modern_kimi_embedding_probe_r461(ctx,epoch,ids,width,pointer,elements));native_wall=monotonic()-native_start
  assert bool((output[:guard]==0x5aa5).all())and bool((output[-guard:]==0x5aa5).all()),'CPU CAPI output boundary canary'
  actual=output[guard:-guard].reshape(8,width,7168);owners=[]
  for device in range(8):
   value=actual[device];score=metrics(np,value,reference)
   # Bit differences are diagnostic. Finite/relative error are the numerical gates.
   assert score['NRMS']<=configuration['numerical_thresholds']['NRMS_max'],score
   assert score['maxabs_over_max1_reference']<=configuration['numerical_thresholds']['maxabs_over_max1_reference'],score
   artifact=store('output-q%d-rank%d.bf16'%(epoch,node*8+device),value.tobytes(order='C'))
   owners.append(dict(rank=node*8+device,device=device,score=score,artifact=artifact))
  total_tokens+=width;actual_census,_=census(lib,node,epoch,total_tokens,backings)
  finished=dict(epoch=epoch,tokens=width,IDs_SHA256=ID_SHA(ids_list),owners=owners,census=actual_census,
   CPU_output_canary_passed=True,memory_after=snapshot('memory_after_query%d'%epoch),native_inclusive_s=native_wall,
   native_clock_scope='host CAPI wall: actual owner scheduling/ID H2D/masked lookup/one TP32 BF16sum/fence/output D2H/CPU copy/all8 waits; no kernel-only claim',
   whole_query_inclusive_s=monotonic()-started,whole_query_clock_scope='local perf_counter: reference/store, memory snapshots, Native inclusive, finite/error checks and all raw output stores')
  results.append(finished);record('lookup_after',finished)
 final=read(lib,'modern_kimi_engine_output_metrics_r2475',ctx)
 assert all(final[k]==0 for k in ['engine_calls','full_logits_calls','token_only_calls','forward_logits_bytes','forward_id_bytes'])
 record('full_Model_forward_metrics_remain_zero',final)
 return dict(passed=True,queries=results,lookup_calls_per_owner=4,total_tokens_per_owner=578,
  weight_keys=1,full_Model_forward=0,HTTP=0,invalid_global_ID_CPU_rejections=2,measured_Model_speed_or_quality=False)
