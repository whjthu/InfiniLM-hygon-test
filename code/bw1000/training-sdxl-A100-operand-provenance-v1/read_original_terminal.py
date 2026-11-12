from pathlib import Path
import hashlib,json,math,os,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-sdxl-original-operand-provenance-v1');started=time.monotonic();assert time.time()<READER_DEADLINE

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p):
 try:st=Path('/proc',str(p['PID']),'stat').read_text().split(') ',1)[1].split()
 except FileNotFoundError:return dict(original_alive=False,state='MISSING')
 return dict(original_alive=int(st[19])==p['startticks'],state='MATCH' if int(st[19])==p['startticks'] else 'REUSED')
intent=json.loads((W/'intent.json').read_bytes());out=json.loads((W/'outcome.json').read_bytes());assert out['claim']==intent['claim']==CLAIM and out['request_id']==intent['request_id']==REQUEST and intent['epoch']==out['epoch'] and intent['deadline']==out['deadline']==out['epoch']+900;assert out['wall_seconds']<900 and not out['forced_cleanup'] and out['passed'] and out['post_clean']
source_pins={}
for n in REMOTE_FILES:
 p=W/'source'/n;assert p.is_file() and not p.is_symlink() and sha(p)==SOURCES[n]==intent['source_files'][n];source_pins[n]=sha(p)
assert len(source_pins)==20;fm=json.loads((W/'source/FRAMEWORK_SUBSET.json').read_bytes());assert fm['total_framework_files']==334 and len(fm['files'])==8
for n,pin in fm['files'].items():
 p=W/'source/framework'/n;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
objects={}
for name,filename,offset in [('CPU','CPU-result.json',120),('GPU','GPU-result.json',660),('compare','comparison-result.json',780)]:
 phase=out['phases'][name];p=W/filename;assert phase['returncode']==0 and phase['reaped'] and sha(W/(name+'.log'))==phase['log_SHA256'];value=json.loads(p.read_bytes());assert p.stat().st_size==phase['result']['bytes'] and sha(p)==phase['result']['SHA256'] and value==phase['result']['value'] and value['epoch']==out['epoch'] and value['deadline']==out['epoch']+offset;objects[name]=value
cpu,gpu,cmp=objects['CPU'],objects['GPU'],objects['compare'];assert cpu['passed'] and cpu['actual_CPU_tests']==14 and not cpu['CUDA_initialized'];assert gpu['passed'] and gpu['actual_optimizer_updates']==1 and gpu['runtime']['accelerator']=='cuda' and gpu['runtime']['cuda_count']==1;assert cmp['evidence_processed'] and not cmp['CUDA_initialized'] and cmp['numerical_passed']==out['source_numerical_passed']
processes={name:identity(p) for name,p in out['owned_processes'].items()}
for name in ['CPU','GPU']:processes[name+'-python']=identity(objects[name]['owner'])
assert not any(v['original_alive'] for v in processes.values());assert intent['GPU_UUID']==[r for r in out['before']['GPUs'] if int(r[0])==4][0][4];uid=intent['GPU_UUID'];assert not any(z[1]==uid for z in out['before']['processes']+out['post']['processes']);assert all(int(z[3])==0 for z in out['post']['GPUs'] if z[4]==uid)
for n,pin in gpu['providers'].items():
 p=Path(pin['path']);assert p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
summary=json.loads((W/'audit/replay/summary.json').read_bytes());assert sha(W/'audit/replay/observed.pt')==gpu['observed_SHA256']==summary['observed_sha256'];audit=json.loads((W/'audit/backward-audit.json').read_bytes());assert audit['passed'] and audit['diagnostic']==gpu['diagnostic'];assert sha(W/'audit/local-operands.private.pt')==gpu['private_capture']['SHA256'];assert sha(W/'comparison.private.json')==cmp['comparison_SHA256']
deps=json.loads((W/'source/DEPENDENCIES.json').read_bytes());reference=Path(deps['reference']);assert reference.stat().st_size==74257307 and sha(reference)==deps['reference_SHA256'];assert sha(W/'source/policy.json')==deps['policy_SHA256'];assert time.time()<READER_DEADLINE
import torch,numpy as np
assert not torch.cuda.is_initialized();torch.set_num_threads(2)
def forbid(*_a,**_kw):raise RuntimeError('GPU forbidden in original source reader')
torch.cuda._lazy_init=forbid;operands=torch.load(W/'audit/local-operands.private.pt',map_location='cpu',weights_only=True);ref=torch.load(reference,map_location='cpu',weights_only=True);name=gpu['diagnostic']['module']+'.weight';assert set(operands)=={'input','output_gradient','weight_gradient'} and all(x.device.type=='cpu' and x.dtype==torch.bfloat16 for x in operands.values());hashes={n:hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for n,t in operands.items()};assert hashes==gpu['diagnostic']['operand_SHA256']
x=operands['input'].reshape(-1,4).double().numpy();dy=operands['output_gradient'].reshape(x.shape[0],-1).double().numpy();exact=np.einsum('ri,rj->ji',x,dy,optimize=False,dtype=np.float64);rounded=torch.from_numpy(exact).to(torch.bfloat16)
def metric(a,b):
 aa,bb=a.double().numpy().reshape(-1),b.double().numpy().reshape(-1);err=math.sqrt(math.fsum(float(z)*float(z) for z in aa-bb));den=math.sqrt(math.fsum(float(z)*float(z) for z in bb));return dict(difference_l2=err,reference_l2=den,relative_l2=err/den if den else (0.0 if err==0 else None),sign_differences=int(np.count_nonzero(np.sign(aa)!=np.sign(bb))),elements=len(aa),exact_equal=bool(np.array_equal(aa,bb)))
independent=dict(native_vs_local_F64_rounded=metric(operands['weight_gradient'],rounded),native_vs_original_gradient=metric(operands['weight_gradient'],ref['gradients_before_clip'][name]),local_F64_rounded_vs_original_gradient=metric(rounded,ref['gradients_before_clip'][name]))
for n,m in independent.items():
 old=gpu['diagnostic'][n]
 for k in ['difference_l2','reference_l2','relative_l2']:
  assert m[k] is None and old[k] is None or m[k] is not None and old[k] is not None and math.isclose(m[k],old[k],rel_tol=1e-11,abs_tol=1e-15)
 assert all(m[k]==old[k] for k in ['sign_differences','elements','exact_equal'])
assert not torch.cuda.is_initialized() and time.time()<READER_DEADLINE
artifacts={n:dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n)) for n in ['intent.json','outcome.json','CPU-result.json','GPU-result.json','comparison-result.json','CPU-tests.private.txt','before.json','post.json','audit/backward-audit.json','audit/replay/summary.json','audit/replay/observed.pt','audit/local-operands.private.pt','comparison.private.json']};answer=dict(evidence_verified=True,claim=CLAIM,request_id=REQUEST,original_outcome=out,actual_source_CPU_tests=14,actual_A100_updates=1,source_numerical_passed=cmp['numerical_passed'],original_source_files_verified=source_pins,original_artifacts=artifacts,original_processes=processes,independent_NumPy_CPU_F64=independent,full_original_comparison=json.loads((W/'comparison.private.json').read_bytes()),CPU_only_reader=True,CUDA_initialized=False,raw_operand_cross_host_transfer_bytes=0,reader_wall_seconds=time.monotonic()-started,Model_GO=False);print(json.dumps(answer,allow_nan=False))
