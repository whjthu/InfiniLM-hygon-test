"""Exact observed Flash2 shapes;controlled BF16 inputs are not model activations."""
import hashlib,json,math
from pathlib import Path
import time
from .zimage_attention_policy import POLICY_ID,policy_for_config,apply_training_attention_policy,_resolve_provider


def run(output,*,epoch,deadline):
    import torch
    if not all(math.isfinite(v) for v in (epoch,deadline)) or not 0<deadline-epoch<=3600:raise ValueError('invalid SDK epoch')
    anchor=time.monotonic();remaining=deadline-time.time()
    def check():
        if min(deadline-time.time(),remaining-(time.monotonic()-anchor))<=0:raise TimeoutError('original SDK epoch expired')
    if torch.cuda.device_count()!=1:raise ValueError('one actual owned CUDA device required')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    module,_,original,_=_resolve_provider();policy=policy_for_config(dict(attention_backward_policy=POLICY_ID));rows=[]
    generator=torch.Generator(device='cpu');generator.manual_seed(20261113)
    torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    def cpu_sha(v):return hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def invoke(inputs,deterministic):
        check();live=[x.to('cuda').requires_grad_() for x in inputs[:3]];grad=inputs[3].to('cuda')
        torch.cuda.synchronize();started=time.monotonic()
        if deterministic:
            with apply_training_attention_policy(policy,check=check) as ledger:
                y=module.flash_attn_func(*live,dropout_p=0.,causal=False);y.backward(grad);torch.cuda.synchronize()
            assert ledger['forced_calls']==1
        else:
            y=original(*live,dropout_p=0.,causal=False);y.backward(grad);torch.cuda.synchronize()
        wall=time.monotonic()-started
        result=[y.detach().cpu()]+[x.grad.detach().cpu() for x in live]
        if not all(bool(torch.isfinite(v).all()) for v in result) or not all(bool(v.abs().max()>0) for v in result[1:]):raise ValueError('invalid actual SDK output or gradient')
        del y,live,grad;torch.cuda.empty_cache();return result,wall
    for seq in (32,256,288):
        check();shape=(1,seq,30,128);inputs=[(torch.randn(shape,generator=generator)*.1).to(torch.bfloat16) for _ in range(4)]
        stride=(seq*3840,3840,128,1);assert all(tuple(v.stride())==stride for v in inputs)
        free,_=torch.cuda.mem_get_info();assert free>=16*1024**3
        base,basewall=invoke(inputs,False);det=[];walls=[]
        for _ in range(3):
            value,wall=invoke(inputs,True);det.append(value);walls.append(wall)
        forward_equal=all(torch.equal(base[0],v[0]) for v in det)
        repeated_equal=all(all(torch.equal(det[0][k],v[k]) for k in range(4)) for v in det[1:])
        row=dict(shape=list(shape),stride=list(stride),dtype='BF16',causal=False,dropout_p=0.,input_kind='Controlled synthetic finite signed BF16;not captured model activations.',
            input_SHA256=[cpu_sha(v) for v in inputs],forward_default_vs_deterministic_equal=forward_equal,three_deterministic_output_and_gradients_equal=repeated_equal,
            default_vs_deterministic_gradient_max_abs=[float((base[k].float()-det[0][k].float()).abs().max()) for k in (1,2,3)],
            default_call_seconds=basewall,deterministic_call_seconds=walls,passed=forward_equal and repeated_equal)
        rows.append(row)
        with (output/('shape-%d.json'%seq)).open('x') as f:json.dump(row,f,indent=2)
        del base,det,inputs;check()
        if not row['passed']:raise ValueError('deterministic SDK fixture failed')
    result=dict(passed=True,cases=rows,actual_forward_calls=12,actual_backward_calls=12,policy=policy,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_seconds=time.monotonic()-anchor,epoch=epoch,deadline=deadline,actual_native_model_loads=0,actual_native_optimizer_updates=0,formal_performance=False,model_numerical_qualified=False,Model_GO=False)
    with (output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    return result
