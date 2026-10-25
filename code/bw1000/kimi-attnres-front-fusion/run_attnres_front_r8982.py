"""Single-card component only. Requires an explicitly pinned, built DSO; no launch wrapper."""
import argparse, ctypes, hashlib, json, pathlib, time

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--sha256', required=True)
p.add_argument('--output', required=True)
p.add_argument('--cpu-receipt', required=True)
p.add_argument('--cpu-receipt-sha256', required=True)
p.add_argument('--device', type=int, default=0)
a = p.parse_args()
library = pathlib.Path(a.library)
assert hashlib.sha256(library.read_bytes()).hexdigest() == a.sha256
assert not pathlib.Path(a.output).exists()
receipt_path=pathlib.Path(a.cpu_receipt)
assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()==a.cpu_receipt_sha256
receipt=json.loads(receipt_path.read_text())
assert receipt['completed'] and receipt['CPU_only'] and not receipt['GPU'] and not receipt['dlopen']
assert receipt['native']==str(library) and receipt['native_SHA256']==a.sha256
for name,expected in receipt['actual_dependency_SHA256'].items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected,name
# Validate the native pin before importing Torch / loading CUDA / loading the DSO.
import torch
from safetensors import safe_open
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_device(a.device)
model = pathlib.Path('/shared/Kimi-K3')
mapping = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
cfg = json.loads((model/'config.json').read_text())['text_config']
assert cfg['hidden_size'] == 7168
weights, weight_hashes = [], {}
for suffix in ['self_attention_res_norm.weight', 'self_attention_res_proj.weight']:
    name = 'language_model.model.layers.46.' + suffix
    with safe_open(model/mapping[name], framework='pt', device='cpu') as f:
        t = f.get_tensor(name).contiguous()
    assert t.dtype == torch.bfloat16 and t.numel() == 7168
    weight_hashes[name] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
    weights.append(t.to(device='cuda').contiguous())
lib = ctypes.CDLL(str(library))
mapped=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libinfiniops.so' in line})
assert len(mapped)==1 and mapped[0] in receipt['actual_dependency_SHA256'],mapped
lib.modern_attnres_error.restype = ctypes.c_char_p
lib.modern_attnres.argtypes = [ctypes.c_int]+[ctypes.c_size_t]*4+[ctypes.c_float]+[ctypes.c_void_p]*5
lib.modern_attnres_front_mode_r8982.argtypes = [ctypes.c_int]
lib.modern_attnres_front_phase_r8982.argtypes = [ctypes.c_int]
lib.modern_attnres_front_count_r8982.argtypes = [ctypes.c_int,ctypes.c_int]
lib.modern_attnres_front_count_r8982.restype = ctypes.c_ulonglong
lib.modern_attnres_shutdown_r8982.argtypes = [ctypes.c_int]
report = {'hardware':'BW1000','level':'synthetic_attention_component','status':'RUNNING',
          'scope':'M16/history24/cap25/H7168; real layer46 small weights, synthetic operands; no Model qualification',
          'native_SHA256':a.sha256,'weights_SHA256':weight_hashes,'arms':[],
          'math':'Explicit BF16 normalized and score rounding; original current-SDK Softmax and weighted Gemm',
          'difference_policy':'Intermediate/output error diagnostic, no bitexact crossarm gate; final Model tokens remain untested'}
def save():
    q = pathlib.Path(a.output)
    temp = q.with_suffix(q.suffix+'.tmp')
    temp.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode())
    temp.replace(q)
def stats(x,y):
    xf,yf=x.float(),y.float(); d=(xf-yf).abs()
    # Ordered BF16 representation for a signed ULP-distance diagnostic.
    def ordered(t):
        z=t.view(torch.int16).to(torch.int32)&65535
        return torch.where((z&32768)!=0,65535-z,z+32768)
    return {'maxabs':d.max().item(),'relative_RMS':((xf-yf).square().mean().sqrt()/yf.square().mean().sqrt().clamp_min(1e-30)).item(),
            'BF16_ULP_max':(ordered(x)-ordered(y)).abs().max().item(),
            'different_elements':int((x!=y).sum().item())}
saved = {}
try:
    gen=torch.Generator(device='cuda').manual_seed(8982001)
    storage=torch.full((16,25,7168),float('nan'),device='cuda',dtype=torch.bfloat16)
    storage[:,:24].copy_(torch.randn((16,24,7168),generator=gen,device='cuda').mul_(.1).to(torch.bfloat16))
    assert not storage[:,:24].is_contiguous()  # Actual [CAP*H,H,1] residual-bank stride.
    report['valid_bank_stride']=list(storage[:,:24].stride())
    schedule=[(0,m) for m in [0,1]]+[(1,m) for m in [0,1,1,0,1,0,0,1]]+[(2,0)]
    for index,(phase,mode) in enumerate(schedule):
        assert lib.modern_attnres_front_mode_r8982(mode)==0
        assert lib.modern_attnres_front_phase_r8982(phase)==0
        padded=torch.full((16*7168+64,),.3125,device='cuda',dtype=torch.bfloat16)
        out=padded[32:-32].view(16,7168);out.fill_(float('nan'))
        torch.cuda.synchronize()
        start=time.perf_counter()
        rc=lib.modern_attnres(a.device,16,24,25,7168,float(cfg['rms_norm_eps']),
            *[ctypes.c_void_p(t.data_ptr()) for t in [storage,*weights,out]],
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        elapsed=time.perf_counter()-start
        if rc: raise RuntimeError(lib.modern_attnres_error().decode())
        assert bool(torch.isfinite(out).all()), 'STOP_nonfinite'
        assert bool(torch.isnan(storage[:,24]).all()), 'STOP_unused_bank_corruption'
        assert bool((padded[:32]==.3125).all() and (padded[-32:]==.3125).all()), 'STOP_output_canary'
        row={'index':index,'phase':['warm','measure','restore'][phase],'mode':mode,
             'wall_s_including_original_drains':elapsed,'included_in_timing':phase==1,'finite':True,'canary':True,
             'counter':[[int(lib.modern_attnres_front_count_r8982(s,m)) for m in [0,1]] for s in range(3)]}
        if mode in saved:
            row['ownrepeat']=stats(out,saved[mode])
            assert torch.equal(out,saved[mode]), 'STOP_ownrepeat_unstable'
        else: saved[mode]=out.clone()
        if 0 in saved and mode==1: row['vs_reference']=stats(out,saved[0])
        report['arms'].append(row);save()
    assert report['arms'][-1]['counter']==[[1,1],[4,4],[1,0]]
    report['status']='COMPONENT_COMPLETED_NOT_MODEL_QUALIFIED'
except BaseException as e:
    report['status']='FAILED_STOP';report['error']=repr(e);save();raise
finally:
    report['shutdown_entered']=True;save()
    rc=lib.modern_attnres_shutdown_r8982(a.device)
    report['shutdown_returncode']=rc;save()
    if rc: raise RuntimeError(lib.modern_attnres_error().decode())
