"""One real embedding+layer0 TP8 component; original O then N, existing hooks.
Raw/weights stay remote. No model token or TPS claim, no synthetic operands.
"""
import pathlib,json,hashlib,ctypes,threading,concurrent.futures,os,time,struct,re
from kimi_native_state_consumer_r8519 import read_snapshot
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def resolve(p):
 s=str(p);prefix='/public/home/aclsylqidf/hygon-inference-refactor-20261014/'
 return pathlib.Path('/refactor')/s[len(prefix):]if s.startswith(prefix)else pathlib.Path(s)
def check_metadata(m,model):
 index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];headers={}
 assert len(m['weights'])==24 and len(m['inputs']['O'])==1 and len(m['inputs']['N'])==4
 assert m['inputs']['O'][0]==m['inputs']['N'][0] and all(len(v)==512 and all(type(t)is int and 0<=t<163840 for t in v)for values in m['inputs'].values()for v in values)
 for k,meta in m['weights'].items():
  assert index[k]==meta['shard'];p=model/index[k];assert p.is_file()
  if p not in headers:
   with p.open('rb')as f:size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size);headers[p]=json.loads(raw)
  actual=headers[p][k];assert all(actual[v]==meta[v]for v in ['dtype','shape','data_offsets']),k
 p=pathlib.Path('/refactor')/m['CPU_receipt'];assert p.is_file() and sha(p)==m['CPU_receipt_raw_SHA256'];b=json.loads(p.read_text());assert b['CPU_only']and not b['GPU_dlopen']and b['library_SHA256']==m['runtime_library_SHA256'];assert sha(m['runtime_library'])==m['runtime_library_SHA256']
 config=json.loads((model/'config.json').read_text());assert config['text_config']['linear_attn_config']['num_heads']==96
 alogs=[v for k,v in m['weights'].items()if k.endswith('.self_attn.A_log')];assert len(alogs)==1 and alogs[0]['shape']==[128]
 return {'A_log_conversion_contract':{'source':[128],'loaded':[96],'same_as_original_loader':True},'checked24realcheckpointHeaders':True,'CPUreceiptSHA':sha(p),'DSO_SHA256':sha(m['runtime_library']),'new_GPU':False}
def inspect_case(directory,rank,expected=None):
 bounds=[];p=directory/f'stage0-rank{rank}.jsonl';assert p.is_file()
 for line in p.read_text().splitlines():
  v=json.loads(line);assert v['stage']==0 and v['rank']==rank and v['layer']==0 and v['packed_target_only'];assert all(q['nonfinite']==0 for q in v['requests']);bounds.append(v)
 journals=list(directory.glob(f'stage0-rank{rank}-layer0-kda-call*.jsonl'));assert len(journals)==1,journals;native={}
 for line in journals[0].read_text().splitlines():
  rec=json.loads(line);assert rec['nonfinite']==0;raw=read_snapshot(directory,rec)
  if rec['point']in ['input','qkv_projected','gated','output']:
   assert raw.shape[0]==1 and raw.shape[1]in(512,2048);native[rec['point']]={'SHA256':hashlib.sha256(raw[:,:512].tobytes()).hexdigest(),'full_SHA256':sha(directory/rec['file']),'shape':list(raw.shape),'dtype':rec['dtype']}
 assert set(native)=={'input','qkv_projected','gated','output'}
 out={'boundary_records':bounds,'native_target':native}
 if expected:
  ex={v['point']:v for v in expected['expected_boundaries']if v['rank']==rank};actual={v['point']:v for v in bounds};assert ex.keys()==actual.keys(),(ex.keys(),actual.keys())
  out['boundary_exact']={k:actual[k]['requests']==ex[k]['requests']for k in ex};en=next(v for v in expected['expected_KDA']if v['stage']==0 and v['rank']==rank)
  out['native_exact']={k:native[k]['SHA256']==en['intermediates'][k]['SHA256']for k in native};out['faithful']=all(out['boundary_exact'].values())and all(out['native_exact'].values())
 return out
def run(manifest,output,root):
 import torch,infinilm,numpy as np
 from safetensors import safe_open
 torch.set_num_threads(1);assert torch.cuda.device_count()==8
 assert os.environ['INFINILM_DENSE_PREFILL_F32_R2475']=='1'
 for flag in ['INFINILM_KIMI_CCL512_R8603','INFINILM_KIMI_PREFILL512_R8562','INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562']:assert os.environ[flag]=='0'
 m=json.loads(pathlib.Path(manifest).read_text());model=pathlib.Path('/shared/Kimi-K3');proof=check_metadata(m,model);root=pathlib.Path(root);root.mkdir(exist_ok=False);path=pathlib.Path(output);assert not path.exists()
 result={'scope':m['scope'],'source_manifest_SHA256':sha(manifest),'proof':proof,'phase':'loading24realweights','workers':[{'rank':i,'normal_close':False}for i in range(8)],'performance':'N/A_CAUSAL_DIAGNOSTIC','final_tokens':'NOT_MEASURED','raw_storage':'REMOTE_ONLY','snapshot_byte_upper':m['snapshot_byte_upper'],'all8normal':False};lock=threading.Lock()
 def save():
  with lock:tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
 save();weights=[];config=(model/'config.json').read_text();cfg=json.loads(config);assert cfg['text_config']['linear_attn_config']['num_heads']==96
 for k,v in sorted(m['weights'].items()):
  with safe_open(model/v['shard'],framework='pt',device='cpu')as f:w=f.get_tensor(k)
  orig_shape=list(w.shape)
  if k.endswith('.self_attn.A_log'):assert orig_shape==[128];w=w[:96]
  w=w.contiguous();name=k.removeprefix('language_model.model.');assert name=='embed_tokens.weight'or name.startswith('layers.0.');weights.append((name,w));
  result.setdefault('weight_sources',[]).append({'checkpoint_key':k,'native_key':name,'source_shape':orig_shape,'loaded_shape':list(w.shape),'dtype':str(w.dtype),'source_shard':v['shard'],'loaded_SHA256':hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()})
 lib=ctypes.CDLL(m['runtime_library']);lib.kimi_layer0_error_r8634.restype=ctypes.c_char_p
 lib.kimi_layer0_id_size_r8634.restype=ctypes.c_size_t;lib.kimi_layer0_id_r8634.argtypes=[ctypes.c_void_p]
 lib.kimi_layer0_create_r8634.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_void_p];lib.kimi_layer0_create_r8634.restype=ctypes.c_void_p
 lib.kimi_layer0_load_r8634.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int]
 for n in ['finish','destroy']:getattr(lib,'kimi_layer0_'+n+'_r8634').argtypes=[ctypes.c_void_p]
 lib.kimi_layer0_prefill_r8634.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
 for name in ['modern_kimi_prefill512_counts_r8562','modern_kimi_CCL512_counts_r8603']:
  getattr(lib,name).argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
 def counters(rank):
  result={}
  for name in ['modern_kimi_prefill512_counts_r8562','modern_kimi_CCL512_counts_r8603']:
   out=(ctypes.c_uint64*3)();assert getattr(lib,name)(rank,out,3)==0;result[name]=list(out)
  return result
 def check(code):
  if code:raise RuntimeError(lib.kimi_layer0_error_r8634().decode())
 uid=(ctypes.c_ubyte*lib.kimi_layer0_id_size_r8634())();check(lib.kimi_layer0_id_r8634(uid))
 barrier=threading.Barrier(8,timeout=180);ready=[False]*8
 for policy in ['Oa','Na','Nb','Nca','Ncb','Ob']:(root/policy).mkdir()
 for k,v in {'INFINILM_KIMI_TRACE_BOUNDARY_LAYERS_R8506':'0','INFINILM_KIMI_TRACE_BOUNDARY_POINTS_R8506':'entry,residual_slot_0,attn_res,attn_norm,attn_output,attn_added,ffn_res,ffn_norm,ffn_output,exit','INFINILM_KIMI_TRACE_HASH_ONLY_R8506':'0','INFINILM_KIMI_TRACE_PACKED_REQUEST_R8508':'0','INFINILM_KIMI_KDA_TRACE_LAYER':'0','INFINILM_KIMI_TRACE_LAYERS':'0','INFINILM_KIMI_TRACE_FULL_FIRST':'0','INFINILM_KIMI_TRACE_FULL_LAST':'0','INFINILM_KIMI_TRACE_STATE_ROWS_MODE_R8516':'all','INFINILM_KIMI_TRACE_KV_POOL_R8511':'0'}.items():os.environ[k]=v
 os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/'Oa')
 def worker(rank):
  torch.cuda.set_device(rank);ctx=None;own=result['workers'][rank]
  try:
   ctx=lib.kimi_layer0_create_r8634(config.encode(),rank,uid);assert ctx,lib.kimi_layer0_error_r8634().decode()
   for key,w in weights:
    shape=(ctypes.c_size_t*w.ndim)(*w.shape);check(lib.kimi_layer0_load_r8634(ctx,key.encode(),w.data_ptr(),w.ndim,shape,{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]))
   check(lib.kimi_layer0_finish_r8634(ctx));own['loaded24']=True;save();barrier.wait()
   for label in ['Oa','Na','Nb','Nca','Ncb','Ob']:
    if rank==0:
     os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/label);result['phase']='capture-'+label
     for flag in ['INFINILM_KIMI_CCL512_R8603','INFINILM_KIMI_PREFILL512_R8562','INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562']:os.environ[flag]='1'if label.startswith('Nc')else'0'
     save()
    barrier.wait();before=counters(rank);ins=np.array(m['inputs'][label[0]],dtype=np.int64);rows=np.arange(1,len(ins)+1,dtype=np.int32);out=np.empty((1,ins.size,7168),dtype=np.uint16);start=time.perf_counter();check(lib.kimi_layer0_prefill_r8634(ctx,len(ins),rows.ctypes.data,ins.ctypes.data,out.ctypes.data));own[label]={'capture_CAPI_wall_s':time.perf_counter()-start,'host_output_target_SHA256':hashlib.sha256(out[:,:512].tobytes()).hexdigest()};after=counters(rank);own[label]['counter_delta']={k:[a-b for a,b in zip(after[k],before[k])]for k in before};save();barrier.wait()
    if label.startswith('Nc'):assert own[label]['counter_delta']=={'modern_kimi_prefill512_counts_r8562':[5,10240,20],'modern_kimi_CCL512_counts_r8603':[1,2048,4]}
    else:assert all(v==[0,0,0]for v in own[label]['counter_delta'].values())
    own[label]['observations']=inspect_case(root/label,rank,m if label[0]=='O'else None)
    if label[0]=='O':ready[rank]=own[label]['observations']['faithful']
    else:
     a=own['Oa']['observations'];b=own[label]['observations'];own[label]['target_changes']={'native':{k:a['native_target'][k]['SHA256']!=b['native_target'][k]['SHA256']for k in a['native_target']},'boundaries':{v['point']:v['requests']!=next(x['requests']for x in a['boundary_records']if x['point']==v['point'])for v in b['boundary_records']}}
    save();barrier.wait()
    if label=='Oa'and not all(ready):own['N_skipped']='Original O notfaithful; do not interpret N';break
   if all(k in own for k in ['Oa','Na','Nb','Nca','Ncb','Ob']):
    own['samegeometry_repeats']={}
    for left,right in [('Oa','Ob'),('Na','Nb'),('Nca','Ncb')]:
     a=own[left]['observations'];b=own[right]['observations'];own['samegeometry_repeats'][left+'-'+right]={'native':{k:a['native_target'][k]['SHA256']==b['native_target'][k]['SHA256']for k in a['native_target']},'boundaries':{v['point']:v['requests']==next(z['requests']for z in b['boundary_records']if z['point']==v['point'])for v in a['boundary_records']}}
    save()
  except BaseException as e:own['error']=repr(e);barrier.abort();save()
  finally:
   if ctx:own['normal_close']=lib.kimi_layer0_destroy_r8634(ctx)==0
   save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(worker,range(8)))
 for flag in ['INFINILM_KIMI_CCL512_R8603','INFINILM_KIMI_PREFILL512_R8562','INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562']:os.environ[flag]='0'
 result.update(phase='completed',all8normal=all(w['normal_close']for w in result['workers']),O_faithful=all(ready),snapshot_bytes=sum(p.stat().st_size for p in root.rglob('*')if p.is_file()));save();assert result['snapshot_bytes']<=m['snapshot_byte_upper'];assert result['all8normal']and not any('error'in w for w in result['workers'])
if __name__=='__main__':
 import sys
 run(*sys.argv[1:])
