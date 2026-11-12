"""Independent CPU-only reference using newly captured local operands."""
from pathlib import Path
import hashlib,json,math,os,runpy,signal,sys,time
W=Path('/t06-run');intent=json.loads((W/'intent.json').read_bytes());remaining=min(110,intent['deadline']-time.time()-60);assert remaining>0
def expired(*_):raise TimeoutError('original comparison budget exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
sys.path.insert(0,str(W/'source/tools'));sys.argv=['hygon_sdxl_compare_first_step.py','--reference','/reference/reference.pt','--policy','/reference/policy.json','--observed',str(W/'candidate/replay/observed.pt'),'--receipt',str(W/'candidate/replay/summary.json'),'--output',str(W/'comparison.json')]
code=0
try:runpy.run_path(str(W/'source/tools/hygon_sdxl_compare_first_step.py'),run_name='__main__')
except SystemExit as e:code=e.code
assert code in (0,1)
import numpy as np
import torch
assert not torch.cuda.is_initialized()
def forbid(*_a,**_kw):raise RuntimeError('GPU forbidden in local operand reader')
torch.cuda._lazy_init=forbid
audit=json.loads((W/'candidate/backward-audit.json').read_bytes());assert audit['passed'] and all(v==1 for v in audit['diagnostic']['counts'].values()) and not audit['numerical_repair_applied'];p=W/'candidate/local-operands.private.pt';raw=p.read_bytes();assert len(raw)==audit['private_capture']['bytes'] and len(raw)<17*1024**2 and hashlib.sha256(raw).hexdigest()==audit['private_capture']['SHA256']
capture=torch.load(p,map_location='cpu',weights_only=True);assert set(capture)=={'input','output_gradient','weight_gradient'} and all(x.device.type=='cpu' and x.dtype==torch.bfloat16 and bool(torch.isfinite(x).all()) for x in capture.values());a=capture['input'];dy=capture['output_gradient'];native=capture['weight_gradient'];assert a.shape[-1]==native.shape[-1]==4 and 0<a.numel()//4<=4096 and dy.shape[-1]==native.shape[0]<=2048
x=a.reshape(-1,4).double().numpy();y=dy.reshape(-1,native.shape[0]).double().numpy();exact=np.einsum('no,nr->or',y,x,optimize=False);absolute=np.einsum('no,nr->or',np.abs(y),np.abs(x),optimize=False);rounded=torch.from_numpy(exact).to(torch.bfloat16);original=torch.load('/reference/reference.pt',map_location='cpu',weights_only=True)['gradients_before_clip'][audit['module']+'.weight'];report=audit['diagnostic']
def metrics(observed,expected):
 a=observed.double().numpy();b=expected.double().numpy();difference=float(np.linalg.norm(a-b));denominator=float(np.linalg.norm(b));return dict(difference_l2=difference,reference_l2=denominator,relative_l2=difference/denominator if denominator else (0. if not difference else None),sign_differences=int(np.count_nonzero(np.sign(a)!=np.sign(b))),elements=int(a.size),exact_equal=bool(np.array_equal(a,b)))
computed={'native_vs_local_F64_rounded':metrics(native,rounded),'native_vs_original_gradient':metrics(native,original),'local_F64_rounded_vs_original_gradient':metrics(rounded,original)}
for name,values in computed.items():
 for key,value in values.items():
  if type(value) is float:assert math.isclose(value,report[name][key],rel_tol=1e-10,abs_tol=1e-15)
  else:assert value==report[name][key]
nonzero=np.abs(exact)>0;zeros=int(np.count_nonzero((~nonzero)&(absolute>0)));ratio=absolute[nonzero]/np.abs(exact[nonzero]);bins={str(bound):int(np.count_nonzero(ratio>=bound))+zeros for bound in (10,100,1000,10000)};assert bins==report['cancellation_ratio_at_least'] and zeros==report['exact_zero_with_nonzero_absolute_sum']
result=dict(passed=True,method='Independent NumPy CPUF64 einsum with optimize=False, then BF16 rounding and scalar comparisons; no native backward replay',module=audit['module'],computed=computed,cancellation_ratio_at_least=bins,exact_zero_with_nonzero_absolute_sum=zeros,captured_tensor_bytes=sum(x.numel()*x.element_size() for x in capture.values()),private_capture_SHA256=hashlib.sha256(raw).hexdigest(),GPU_initialized=False,GPU_cards=0,original_comparator_returncode=code,Model_GO=False)
with (W/'backward-independent.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(independent_operand_reference_passed=True,original_numerical_comparator_returncode=code,GPU_cards=0)))
raise SystemExit(code)
