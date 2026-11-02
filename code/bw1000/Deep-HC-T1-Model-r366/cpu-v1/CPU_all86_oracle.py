"""Independent CPU full-HC oracle on every real HC parameter group; no Native/GPU/Model."""
import pathlib,json,hashlib,time,os,re,numpy as np
from reference import full_pre,bf16_rn
M=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
R=pathlib.Path('/refactor');B=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
bits_sha=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
started=time.perf_counter()
import torch
from safetensors import safe_open
assert not torch.cuda.is_initialized()
cfg=json.loads((M/'config.json').read_bytes());index=json.loads((M/'model.safetensors.index.json').read_bytes())['weight_map']
assert cfg['num_hidden_layers']==43 and cfg['hidden_size']==4096
groups=[(int(m.group(1)),m.group(2))for k in index for m in [re.fullmatch(r'layers\.(\d+)\.hc_(attn|ffn)_fn',k)]if m]
groups.sort();assert len(groups)==86 and set(groups)=={(l,k)for l in range(43)for k in ['attn','ffn']}
oldpath=R/'candidates/Deep-HC-epilogue-r358-cpu-r2/CPU-fixtures.json';assert sha(oldpath)=='e282795fd51f067b49919fc17cd9d3450fda76039e0b4c664422e74c5d63cc00';old=json.loads(oldpath.read_bytes());assert old['passed']
rows=[];reused=0
for layer,kind in groups:
    arrays=[];weights=[]
    for name,shape in [('fn',[24,16384]),('base',[24]),('scale',[3])]:
        key=f'layers.{layer}.hc_{kind}_{name}';assert key in index
        with safe_open(M/index[key],framework='pt',device='cpu')as f:t=f.get_tensor(key).contiguous()
        assert t.dtype==torch.float32 and list(t.shape)==shape
        a=t.numpy().copy();assert np.isfinite(a).all();arrays.append(a);weights.append(dict(key=key,shape=shape,dtype='F32',SHA256=bits_sha(a),shard=index[key]))
    state=[bits_sha(a)for a in arrays];cases=[]
    for seed,std in [(781,.5),(1889,2.)]:
        rng=np.random.default_rng(seed);x=rng.standard_normal((1,4,4096),dtype=np.float32)*np.float32(std);bits=bf16_rn(x)
        expected=full_pre(bits,*arrays,float(cfg['rms_norm_eps']),float(cfg['hc_eps']),int(cfg['hc_sinkhorn_iters']))
        assert all(np.isfinite(v).all()for v in expected.values());collapsed=expected['collapsed_BF16'];assert np.isfinite((collapsed.astype(np.uint32)<<16).view(np.float32)).all()
        assert state==[bits_sha(a)for a in arrays]
        outputs={n:bits_sha(v)for n,v in expected.items()};prior=[p for p in old['fixtures']if p['layer']==layer and p['kind']==kind and p['seed']==seed]
        if prior:
            assert len(prior)==1 and prior[0]['weights']==weights and prior[0]['input_SHA256']==bits_sha(bits)and prior[0]['CPU_reference_output_SHA256']==outputs;reused+=1
        cases.append(dict(seed=seed,std=std,input_BF16_SHA256=bits_sha(bits),CPU_reference_output_SHA256=outputs,finite=True,state_unchanged=True,prior_CPU700_same_group_verified=bool(prior)))
    rows.append(dict(layer=layer,kind=kind,weights=weights,weight_state_initial_SHA256=state,weight_state_final_SHA256=[bits_sha(a)for a in arrays],cases=cases))
assert reused==8 and not torch.cuda.is_initialized()
maps=pathlib.Path('/proc/self/maps').read_text();assert 'libDeep_latest_native_r269.so'not in maps
q=dict(passed=True,CPU_only=True,GPU_initialized=False,Native_dlopen=False,whole_Target_Model_loaded=False,small_HC_parameter_tiles_only=True,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],config_SHA256=sha(M/'config.json'),index_SHA256=sha(M/'model.safetensors.index.json'),actual_HC_groups=86,oracle_cases=172,prior_CPU700_cases_verified=8,prior_CPU700_fixture_SHA256=sha(oldpath),groups=rows,seconds=time.perf_counter()-started,scope='Independent CPU fullprefix/F32Gemm/affine/Sigmoid/Softmax/originalSinkhorn/BF16rounding oracle, actual43layersx2HCparametersx2syntheticBF16seeds. No newcomponentGPU orModelquality/speed claim.')
(B/'CPU-all86-oracle.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(dict(passed=True,HC_groups=86,cases=172,reused_prior8=reused,seconds=q['seconds'])))
