import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import ctypes,json,pathlib,time,hashlib,datetime,socket
import torch,infinilm
from stage_control_r2386 import StageControl
torch.set_num_threads(1)
assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
r=pathlib.Path('/refactor');rank=int(os.environ['SLURM_PROCID']);run=os.environ['GLOBAL_RUN_R2354']
p=r/'results'/f'kimi-cap128-3page-cold-r8669aaf-{run}-node{rank}.json';assert not p.exists()
q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'node':socket.gethostname(),'run':run,'completed':False,'weight_load_calls':0,'finish_calls':0,'forward_calls':0,'workspace_cap':64,'snapshots':[],'scope':'ExistingCAPI cap1fourpageconstructor -> setpages3 -> resize128 -> originalreset. NOT directfirstconstructor3page; oldtinycache/poolretention peak measured, not guessed away. No weightload/finish/forward/Modelqualification.'}
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
def memory():
 for d in range(8):torch.cuda.synchronize(d)
 return [list(torch.cuda.mem_get_info(d)) for d in range(8)]
control=StageControl(rank,os.environ['KIMI_ENGINE_HEAD'],int(os.environ['GLOBAL_PORT_R2354'])+7000,os.environ['KIMI_ENGINE_NONCE'],4,180)
ctx=None
try:
 assert os.environ['INFINILM_KIMI_MOE_WORKSPACE_R4665']=='1' and os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665']=='0'
 libpath=r/'candidates/Kimi-CCL512-r8606/libKimi_CCL512_r8606.so'
 q['library_sha256']=hashlib.sha256(libpath.read_bytes()).hexdigest();assert q['library_sha256']=='f6f10c2b243e0336074b872859e6ca6acb08effdcd56d86b9ac4df1bf2ddb5e9'
 lib=ctypes.CDLL(str(libpath));lib.modern_kimi_engine_error_r2475.restype=ctypes.c_char_p
 def check(code):
  if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 lib.modern_kimi_engine_create_r2475.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_size_t];lib.modern_kimi_engine_create_r2475.restype=ctypes.c_void_p
 for name in ['reset','destroy']:getattr(lib,'modern_kimi_engine_'+name+'_r2475').argtypes=[ctypes.c_void_p]
 lib.modern_kimi_engine_resize_cache_r2475.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 for name in ['cache_geometry','layout','output_metrics']:
  f=getattr(lib,'modern_kimi_engine_'+name+'_r2475');f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_char_p
 cfg=json.loads(pathlib.Path('/shared/Kimi-K3/config.json').read_text())
 q['config_SHA256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest();q['before_create_memory']=memory();assert min(x[0] for x in q['before_create_memory'])>58*1024**3;save()
 control.collect('before-create',{'rank':rank});start=time.perf_counter()
 ctx=lib.modern_kimi_engine_create_r2475(json.dumps(cfg).encode(),rank,1,4,os.environ['KIMI_ENGINE_HEAD'].encode(),int(os.environ['GLOBAL_PORT_R2354']),1)
 if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 q['create_seconds']=time.perf_counter()-start;q['layout']=json.loads(lib.modern_kimi_engine_layout_r2475(ctx))
 assert q['layout']['local_workers']==8 and q['layout']['PP']==4 and q['layout']['TP']==8
 lib.modern_kimi_engine_set_cache_pages_r4974.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 initial_mem=memory();q['snapshots'].append({'phase':'initial4pages_cap1','geometry':json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx)),'memory':initial_mem});save();assert all(x[0]>3*1024**3 for x in initial_mem),'original3GiB_aftercreate_reserve'
 check(lib.modern_kimi_engine_set_cache_pages_r4974(ctx,3))
 control.collect('selected3pages_atcap1',{})
 q['snapshots'].append({'phase':'selected3pages_cap1','geometry':json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx)),'memory':memory()});save()
 check(lib.modern_kimi_engine_resize_cache_r2475(ctx,128))
 check(lib.modern_kimi_engine_reset_r2475(ctx))
 geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx))
 assert geometry=={'max_rows':128,'pages':387,'page_size':256,'KDA_pool_rows':129,'pages_per_request':3,'max_total_tokens':768}
 localmem=memory();q['snapshots'].append({'phase':'cap128_threepages_after_existingreset','geometry':geometry,'memory':localmem});save()
 packets=control.collect('actual-cap128-3page-memory',{'memory':localmem})
 q['global_memory_gate']={'minimum_free_bytes':min(x[0]for p in packets for x in p['memory']),'required_free_bytes':4*1024**3};save()
 assert q['global_memory_gate']['minimum_free_bytes']>4*1024**3,'original4GiBreserve_no_lowering'
 q['metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));assert q['metrics']['engine_calls']==0
 control.collect('before-destroy',{});q['allocation_checks_passed']=True
except BaseException as e:q['error']=repr(e);save();raise
finally:
 try:
  if ctx:check(lib.modern_kimi_engine_destroy_r2475(ctx));q['all8_owned_workers_shutdown_completed']=True
  if q.get('allocation_checks_passed'):control.collect('after-destroy',{})
  q['after_destroy_memory']=memory();q['completed']=bool(q.get('allocation_checks_passed') and q.get('all8_owned_workers_shutdown_completed'))
 except BaseException as e:q['shutdown_error']=repr(e);q['completed']=False;raise
 finally:save();control.close()
print('KIMI_CAPACITY_DRY_COMPLETE',rank,flush=True)
