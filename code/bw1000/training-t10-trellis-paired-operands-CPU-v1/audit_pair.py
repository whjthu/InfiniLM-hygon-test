"""Actual saved A100/HIP LoRA operand counterfactuals on CPU only."""
from pathlib import Path
import hashlib,importlib.util,json,os,resource,signal,socket,sys,time,traceback,unittest
epoch=float(os.environ['PAIR_EPOCH']);deadline=float(os.environ['PAIR_DEADLINE']);assert deadline==epoch+180 and epoch<=time.time()<deadline
signal.setitimer(signal.ITIMER_REAL,deadline-time.time());started=time.monotonic();source=Path('/stage-source');out=Path('/evidence')
assert socket.gethostname().split('.')[0]=='a14r2n02' and os.environ['SLURM_JOB_ID']=='1087656'
assert all(os.environ[k]=='-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
sys.path.insert(0,str(source));import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from gencuda.trellis_operand_attribution import attribute_sites
owner=dict(node='a14r2n02',step='1087656.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]));stage='unit_tests';plan=json.loads((source/'PLAN.json').read_bytes())
def progress(complete=False,failed=False):
 p=out/'progress.tmp';p.write_text(json.dumps(dict(owner=owner,stage=stage,planned_sites=4,completed_sites=4 if complete else 0,failed=failed)));p.replace(out/'progress.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):
   assert time.time()<deadline;h.update(raw)
 return h.hexdigest()
try:
 progress();spec=importlib.util.spec_from_file_location('test_actual_attribution',source/'test_attribution.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
 tested=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(tests.OperandAttributionTest));assert tested.testsRun==3 and tested.wasSuccessful()
 stage='authenticate_saved_operand_files';progress()
 a100=Path('/a100-operands.pt');hip=Path('/hip-operands.pt')
 for p,pin in ((a100,plan['a100']),(hip,plan['hygon'])):assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 a=torch.load(a100,map_location='cpu',weights_only=True);h=torch.load(hip,map_location='cpu',weights_only=True)
 sites=sorted(plan['a100']['sites']);assert len(sites)==4 and set(a)==set(sites) and set(h)=={x+'.lora_B' for x in sites}
 def tensor_pin(t):return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
 for site in sites:
  for field in ('input','upstream','native_gradient'):
   tensor=a[site][field];pin=plan['a100']['sites'][site][field];assert tensor.device.type=='cpu' and list(tensor.shape)==pin['shape'] and str(tensor.dtype)==pin['dtype'] and tensor_pin(tensor)==pin['SHA256']
 reference={s:dict(input=a[s]['input'],upstream=a[s]['upstream'],gradient=a[s]['native_gradient']) for s in sites}
 candidate={s:dict(input=h[s+'.lora_B']['input'],upstream=h[s+'.lora_B']['upstream'],gradient=h[s+'.lora_B']['native_grad']) for s in sites}
 stage='actual_four_site_attribution';progress();analysis=attribute_sites(reference,candidate);assert not torch.cuda.is_initialized();progress(True)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,CPU_tests=3,framework_commit=plan['framework']['commit'],module_SHA256=sha(source/'gencuda/trellis_operand_attribution.py'),a100_file_SHA256=plan['a100']['SHA256'],hygon_file_SHA256=plan['hygon']['SHA256'],analysis=analysis,wall_seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=str(torch.__version__),GPU_initialized=False,GPU_cards=0,model_replayed=False,full_adapter_qualified=False,numerical_accepted=False,formal_performance=False,Model_GO=False)
except BaseException as error:
 progress(failed=True);(out/'failure.private.txt').write_text(traceback.format_exc());result=dict(passed=False,owner=owner,stage=stage,error_type=type(error).__name__,message=str(error)[:300],planned_sites=4,wall_seconds=time.monotonic()-started,GPU_cards=0,Model_GO=False)
finally:
 with (out/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
