"""Actual new component entry through ctypes/libffi, recording Native/source leaves.

No Native SDK/dlopen/checkpoint payload or GPU. Does not rerun old CPU suites.
"""
from pathlib import Path
from types import SimpleNamespace
import ctypes as C,hashlib,json,tempfile,time,sys
import numpy as np
import constructor,embedding
class Source:
 descriptor=dict(name=embedding.KEY,pointer=0x123456789abc0000,shape=[163840,7168],dtype=1)
 metadata=dict(CPU_tensor_bytes=2348810240,actual_checkpoint_payload=False,recording_source_only=True)
 @staticmethod
 def reference(ids):
  values=(np.asarray(ids,dtype=np.float32)[:,None]%32+np.arange(7168,dtype=np.float32)[None,:]%32/64+0.5)/32
  return(values.view(np.uint32)>>16).astype(np.uint16)
def metadata():
 return dict(model_type='kimi_k3',text_config=dict(num_hidden_layers=93,hidden_size=7168,num_attention_heads=96,qk_nope_head_dim=128,kv_lora_rank=512,qk_rope_head_dim=64,max_position_embeddings=32768,
  linear_attn_config=dict(num_heads=96,head_dim=128),quantization_config=dict(format='mxfp4-pack-quantized',quant_method='compressed-tensors',config_groups={'group_0':{'weights':dict(group_size=32,num_bits=4,scale_dtype='torch.uint8',type='float',symmetric=True,dynamic=False)}})))
def observations(node):
 owners=[]
 for d in range(8):
  rank=node*8+d;owners.append(dict(device=d,TP=32,PP=1,PP_stage=0,tp_rank=rank,model_constructed=True,head_constructed=True,physical_head_shape=[5120,7168],head_parameter_bytes=73400320,global_row_begin=rank*5120,global_row_end_exclusive=(rank+1)*5120,full_GPU_head_retained=False,head_is_ReplicatedLinear=False,postload_rows_verified=False,head_forward_calls=0,head_forward_rows=0))
 return dict(capabilities=dict(version=1,TP=32,PP=1,logical_capacity=1,physical_rows=1,state_pool_rows=2,context=32768,page_size=256,pages_per_request=128,num_blocks=132,MLA_shape=[2,132,256,3,192],W2_exact_parameter=True,MLA_exact_parameter=True,capture=False,replay=False,GPU_qualified=False),
  layout=dict(node=node,local_workers=8,TP=32,TP_node_rank=node,PP=1,PP_stage=0,sampling_node=0,public_graph_enabled=False,owned_metadata_enabled=False),
  cache_geometry=dict(max_rows=1,pages=132,page_size=256,KDA_pool_rows=2,pages_per_request=128,max_total_tokens=32768),keys=['language_model.model.layers.%d.weight'%i for i in range(93)]+[embedding.KEY,'language_model.lm_head.weight'],
  head=dict(head_kind='ShardOnlyVocabHeadR9183',expected_CPU_source_registered=False,owners=owners),affinity=dict(owners=[dict(device=d,changes=0,verified_mask='')for d in range(8)]),
  output_metrics=dict(engine_calls=0,full_logits_calls=0,token_only_calls=0,forward_logits_bytes=0,forward_id_bytes=0))
def leaves(node,fail_query=None):
 lib=SimpleNamespace();keep=[];calls=[];errors=[];ctx=0x12345678abcdef01;state=dict(epoch=0,tokens=0,loaded=False,destroyed=False)
 actual=observations(node)
 def census():
  return dict(TP=32,full_vocabulary=163840,hidden=7168,owners=[dict(device=d,constructed=True,destroyed=state['destroyed'],tp_rank=node*8+d,physical_shape=[5120,7168],parameter_bytes=73400320,registered_weight_aliases_same_tensor=True,global_row_begin=(node*8+d)*5120,global_row_end_exclusive=(node*8+d+1)*5120,
   forward_calls=state['epoch'],BF16_sum_collectives=state['epoch'],completed_fences=state['epoch'],tokens=state['tokens'],postload_checked=bool(state['epoch']),backing_address=0x888800000000+ d*0x10000000)for d in range(8)])
 mapping={'modern_kimi_ar_storage_capabilities_r456':'capabilities','modern_kimi_engine_keys_r2475':'keys','modern_kimi_engine_layout_r2475':'layout','modern_kimi_engine_cache_geometry_r2475':'cache_geometry','modern_kimi_engine_output_metrics_r2475':'output_metrics','modern_kimi_shard_head_info_r9183':'head','modern_owner_affinity_report_r2392':'affinity'}
 def invoke(name,*args):
  calls.append(name)
  if name=='modern_kimi_engine_create_vocab_parallel_r461':
   assert args[1]==node and args[2]==b'10.1.14.28'and args[3]==19743 and json.loads(args[0])==metadata();return ctx
  if name=='modern_kimi_engine_destroy_r2475':assert args[0]==ctx;state['destroyed']=True;return 0
  if name=='modern_kimi_engine_partition_r2475':assert args[:2]==(0,1);args[2][0]=0;args[2][1]=93;return 0
  if name=='modern_kimi_embedding_probe_load_r461':
   assert args[0]==ctx and json.loads(args[1])==[Source.descriptor]and not state['loaded'];state['loaded']=True;return 0
  if name=='modern_kimi_embedding_probe_r461':
   pointer,epoch,ids,width,out,size=args;assert pointer==ctx and size==8*width*7168 and state['loaded']
   values=[ids[i]for i in range(width)]
   if any(v<0 or v>=163840 for v in values):assert epoch==1 and state['epoch']==0;return 1
   assert epoch==state['epoch']+1 and values==embedding.queries()[epoch-1]
   if fail_query==epoch:return 1
   reference=Source.reference(values);outputs=np.tile(reference,(8,1,1)).copy();C.memmove(out,outputs.ctypes.data,size*2)
   state['epoch']=epoch;state['tokens']+=width;return 0
  if name=='modern_kimi_vocab_embedding_info_r461':payload=json.dumps(census()).encode()
  elif name=='modern_kimi_engine_error_r2475':payload=b'recorded Native failure'
  else:
   if args:assert args[0]==ctx,'high64 context truncated'
   payload=json.dumps(actual[mapping[name]]).encode()
  buf=C.create_string_buffer(payload);keep.append(buf);return C.addressof(buf)
 for name,(args,result)in{**constructor.SIGNATURES,**embedding.SIGNATURES}.items():
  callback_result=C.c_void_p if result is C.c_char_p else result
  def call(*a,n=name,r=callback_result):
   try:return invoke(n,*a)
   except BaseException as error:errors.append(repr(error));return 0 if r is C.c_void_p else 1
  cb=C.CFUNCTYPE(callback_result,*args)(call);fn=C.CFUNCTYPE(callback_result,*args)(C.cast(cb,C.c_void_p).value)
  keep.extend([cb,fn]);setattr(lib,name,fn)
 lib.keep=keep;return lib,state,calls,errors
def main():
 d=Path(__file__).resolve().parent;q=json.loads((d/'config.json').read_bytes());cases=[]
 for label,fail,past in [('complete_new_entry',None,False),('deadline_before_any_Native',None,True),('Native_error_stops_before_remaining_lookups',2,False)]:
  lib,state,calls,errors=leaves(3,fail);events=[];storage={};source=Source();dead=time.time()+60 if not past else time.time()-1
  def record(e,p):events.append((e,p))
  def store(name,raw):storage[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return dict(path='recording:'+name,**storage[name])
  failed=False
  if past:
   try:embedding.run(lib,0x12345678abcdef01,source,node=3,configuration=q,record=record,memory=lambda:[],store=store,deadline=dead)
   except AssertionError:failed=True
   assert failed and not calls
  else:
   try:
    keys=observations(3)['keys']
    with constructor.resident_constructor(lib,metadata(),dict.fromkeys(keys,'recording-only'),node=3,address='10.1.14.28',port=19743,record=record)as(ctx,_):
     result=embedding.run(lib,ctx,source,node=3,configuration=q,record=record,memory=lambda:[[12*1024**3,q['physical_VRAM_bytes']]for _ in range(8)],store=store,deadline=dead)
     assert result['passed']and state['epoch']==4 and state['tokens']==578
   except RuntimeError:failed=True
   assert failed==bool(fail)and state['destroyed']
   assert calls.count('modern_kimi_engine_create_vocab_parallel_r461')==calls.count('modern_kimi_engine_destroy_r2475')==1
   assert calls.count('modern_kimi_embedding_probe_load_r461')==1
   assert state['epoch']==(1 if fail else 4)
  assert not errors,errors
  assert not any(name in calls for name in ['modern_kimi_engine_finish_r2475','modern_kimi_engine_forward_ar_storage_r456'])
  cases.append(dict(case=label,passed=True,high64_context_preserved=True,actual_libffi=True,Native_leaf_recording=True,
   actual_Native=False,actual_checkpoint=False,Native_calls=calls.count('modern_kimi_embedding_probe_r461'),successful_lookup_epochs=state['epoch'],raw_recording_bytes=sum(v['bytes']for v in storage.values())))
 output=dict(passed=True,scope='actual component work entry and ctypes/libffi signatures with recording constructor/Native/source leaves; not device qualification',cases=cases,
  Native_dlopen=False,SDK_execution=False,GPU=0,Slurm=0,Model=0,query_ID_SHA256=[embedding.ID_SHA(v)for v in embedding.queries()])
 print(json.dumps(output,indent=2));return output
if __name__=='__main__':
 result=main()
 if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
