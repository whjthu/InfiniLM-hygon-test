"""No GPU/Native: verify actual small parameter tiles and CPU oracle contracts."""
import pathlib,json,hashlib,os,time,traceback,numpy as np
from fixture import load
from reference import full_pre,bf16_rn
R=pathlib.Path('/refactor');B=R/'candidates/Deep-HC-epilogue-r358-cpu-r2';started=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,whole_Model_loaded=False,small_HC_parameter_tiles_only=True,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],fixtures=[])
try:
 import torch
 assert not torch.cuda.is_initialized()
 bits=np.full((1,4,4096),0x3f80,np.uint16);eps=np.float32(1e-5)
 test=full_pre(bits,np.zeros((24,16384),np.float32),np.zeros(24,np.float32),np.ones(3,np.float32),1e-5,float(eps),1)
 assert np.all(test['collapsed_BF16']==0x4000)and np.all(test['post']==1)
 expected=np.float32(np.float32(.25)+eps)/np.float32(np.float32(1)+np.float32(5)*eps)
 assert np.max(np.abs(test['comb']-expected))<1e-7
 assert np.array_equal(bf16_rn(np.array([1+1/256,1+3/256,-1-1/256,-1-3/256],np.float32)),np.array([0x3f80,0x3f82,0xbf80,0xbf82],np.uint16))
 for rank in range(8):
  data,proof=load(rank);assert all(np.isfinite(v).all()for v in data['expected'].values());assert data['expected']['collapsed_BF16'].shape==(4096,);q['fixtures'].append(proof)
 assert not torch.cuda.is_initialized()and 'libDeep_latest_native_r269.so'not in pathlib.Path('/proc/self/maps').read_text()
 q.update(completed=True,passed=True,seconds=time.monotonic()-started,oracle_cases=['uniformzero-mixes-with-known-Sinkhorn','positive-negativeBF16tie-even','8actualparameter-and-inputfixtures'])
except BaseException as exc:q.update(completed=True,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
(B/'CPU-fixtures.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','actual_step','seconds','error']}));assert q['passed']
