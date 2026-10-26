import pathlib,os,json,time,ctypes as C,threading,concurrent.futures,hashlib,datetime,sys
sys.path.append('/refactor/incoming')
from stage_control_r2386 import StageControl
DSO=pathlib.Path('/refactor/candidates/CCL-raw-AR-r9176/libCCL_raw_AR_r9176.so')
EXPECTED='68034efbcfc4976ba8b3f0125e1d024ba27be64eb0474a7079ef828f00407579'
def main():
 assert hashlib.sha256(DSO.read_bytes()).hexdigest()==EXPECTED
 node=int(os.environ['SLURM_PROCID']);assert node in [0,1]
 assert os.environ['NCCL_NET']=='IBext_v8'and os.environ['NCCL_NET_PLUGIN']=='shca'
 assert len(os.sched_getaffinity(0))>=32
 lib=C.CDLL(str(DSO));lib.ccl_ar_error_r9176.restype=C.c_char_p;lib.ccl_ar_id_size_r9176.restype=C.c_size_t;assert lib.ccl_ar_id_size_r9176()==136
 lib.ccl_ar_id_r9176.argtypes=[C.c_void_p];lib.ccl_ar_create_r9176.argtypes=[C.c_int,C.c_int,C.c_void_p];lib.ccl_ar_create_r9176.restype=C.c_void_p
 lib.ccl_ar_prepare_r9176.argtypes=[C.c_void_p,C.c_size_t,C.c_int];lib.ccl_ar_run_r9176.argtypes=[C.c_void_p,C.c_int];lib.ccl_ar_run_r9176.restype=C.c_char_p;lib.ccl_ar_close_r9176.argtypes=[C.c_void_p]
 def ck(rc):
  if rc:raise RuntimeError(lib.ccl_ar_error_r9176().decode())
 control=StageControl(node,'a02r3n17',int(os.environ['CCL_AR_CONTROL_PORT']),os.environ['CCL_AR_NONCE'],2,240)
 local=C.create_string_buffer(136)
 if node==0:ck(lib.ccl_ar_id_r9176(local))
 # Bootstrap identifiers and authentication stay in runtime memory, not result files.
 packets=control.collect('communicator-bootstrap',{'id':local.raw.hex()}if node==0 else{})
 raw=bytes.fromhex(packets[0]['id']);assert len(raw)==136;identifier=C.create_string_buffer(raw,136)
 cpus=sorted(os.sched_getaffinity(0))[:32];barrier=threading.Barrier(8,timeout=180)
 def worker(dev):
  os.sched_setaffinity(0,set(cpus[4*dev:4*dev+4]));rank=node*8+dev;ctx=None;out={'rank':rank,'device':dev,'points':[],'normal':False}
  try:
   ctx=lib.ccl_ar_create_r9176(rank,dev,identifier)
   if not ctx:raise RuntimeError(lib.ccl_ar_error_r9176().decode())
   for dtype in [9,10]:
    for size in [229376,1048576,16777216,67108864]:
     ck(lib.ccl_ar_prepare_r9176(ctx,size,dtype));barrier.wait();b=lib.ccl_ar_run_r9176(ctx,10)
     if not b:raise RuntimeError(lib.ccl_ar_error_r9176().decode())
     q=json.loads(b);assert q['rank']==rank and q['input_readonly']and q['full_output_sum136_wrong_count']==0;out['points'].append(q)
   ck(lib.ccl_ar_close_r9176(ctx));ctx=None;out['normal']=True;return out
  except BaseException:
   barrier.abort();raise
  finally:
   if ctx:ck(lib.ccl_ar_close_r9176(ctx))
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:owners=list(pool.map(worker,range(8)))
  assert len(owners)==8 and all(x['normal']for x in owners);control.collect('allnormal',{'node':node,'owners':8})
  q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'node':node,'host':__import__('socket').gethostname(),'owners':owners,'native_SHA256':EXPECTED,'Model_weights':False,'GPU_F64':False,'normal_owners':8,'arm':os.environ['CCL_AR_ARM'],'channel_env':{k:os.environ.get(k)for k in ['NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS']},'scope':'Raw pinned CCL two-node16rank; stream elapsed includes arrival/wait, not purebusy; network/GDR evidence independent'}
  pathlib.Path('/refactor/results/CCL-raw-AR-r9217-'+os.environ['CCL_AR_ARM']+'-node%d.json'%node).write_text(json.dumps(q,indent=2));print('CCL_RAW_AR_R9217_NORMAL',node,8,flush=True)
 finally:control.close()
if __name__=='__main__':main()
