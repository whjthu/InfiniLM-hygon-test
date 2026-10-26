import ctypes as C,pathlib,json,hashlib,os,socket
p=pathlib.Path('/refactor/candidates/Kimi-speculative-batch-Model-r9122c/libKimi_speculative_batch_Model_r9122c.so');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='9647453d13f5e56b81fda3a6ec699f5329a83157a294e7e8e2dcffeeb4775e61';lib=C.CDLL(str(p));f=lib.modern_kimi_engine_partition_r2475;f.argtypes=[C.c_int,C.c_int,C.POINTER(C.c_size_t)];f.restype=C.c_int
ranges={}
for pp in [2,4]:
 rows=[]
 for stage in range(pp):
  a=(C.c_size_t*2)();assert f(stage,pp,a)==0;rows.append(list(a))
 ranges[str(pp)]=rows
assert ranges['4']==[[0,24],[24,47],[47,71],[71,93]] and ranges['2']==[[0,46],[46,93]]
q={'CPU_only':True,'dlopen':True,'Runtime_created':False,'GPU':False,'weights_read':False,'native_SHA256':h,'actual_partition_CAPI':ranges,'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'completed':True};out=pathlib.Path('/refactor/results/Kimi-actual-partition-CAPI-r9156.json');assert not out.exists();out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
