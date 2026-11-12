"""CPU-only optimizer causality probe using actual recorded BF16 operands.

The same real first-step gradient is deliberately repeated as a controlled
optimizer workload. This is not the model's native gradient sequence, a native
model replay, a training acceptance result, or a performance sample.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))


def run(binding, output, *, epoch, deadline):
    import torch
    from gencuda import hy_resume_checkpoint as common
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only probe forbids an initialized GPU')
    torch.set_num_threads(1)
    if not all(math.isfinite(v) for v in (epoch,deadline)) or not 0<deadline-epoch<=300:
        raise ValueError('invalid CPU probe epoch')
    started=time.monotonic();remaining=deadline-time.time()
    def check():
        if min(deadline-time.time(),remaining-(time.monotonic()-started))<=0:raise TimeoutError('CPU optimizer probe expired')
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as stream:
            while True:
                check();raw=stream.read(1024**2)
                if not raw:break
                h.update(raw)
        return h.hexdigest()
    root=Path(binding['reference_root']);manifest=root/binding['capture_manifest']['relative_path']
    if not manifest.is_file() or manifest.is_symlink() or sha(manifest)!=binding['capture_manifest']['SHA256']:
        raise ValueError('original first-update manifest differs')
    progress=json.loads(manifest.read_bytes())
    if progress['phase']!='NATIVE25_RETURNED' or progress['completed_global_steps']!=25 or not progress['first_update_complete']:
        raise ValueError('expected the actual completed native25 first-update capture')
    pins={p['name']:p for p in progress['files']};values={}
    for name in ('initial.pt','postclip.pt'):
        p=manifest.parent/name;pin=pins[name]
        if p.is_symlink() or p.stat().st_size!=pin['bytes'] or sha(p)!=pin['SHA256']:
            raise ValueError('original recorded optimizer operand differs')
        values[name]=torch.load(p,map_location='cpu',weights_only=True)
    initial=values['initial.pt'];gradient=values['postclip.pt'];names=list(initial)
    if len(names)!=476 or set(initial)!=set(gradient) or sum(t.numel() for t in initial.values())!=9922560:
        raise ValueError('full actual trainable extent differs')
    for name,tensor in initial.items():
        g=gradient[name]
        if (tensor.device.type!='cpu' or tensor.dtype!=torch.bfloat16 or not tensor.is_contiguous()
                or g.device.type!='cpu' or g.dtype!=tensor.dtype or g.shape!=tensor.shape or not g.is_contiguous()
                or not bool(torch.isfinite(tensor).all()) or not bool(torch.isfinite(g).all())):
            raise ValueError('expected finite actual contiguous CPU BF16 values')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    results=[];actual_updates=0
    def report(phase,**kwargs):
        value=dict(phase=phase,planned_foreach_cases=[None,True],planned_control_steps=25,planned_restored_steps=5,
                   actual_optimizer_steps=actual_updates,completed_cases=results,epoch=epoch,deadline=deadline,**kwargs)
        p=output/'progress.tmp';p.write_text(json.dumps(value,indent=2));p.replace(output/'progress.json')
    report('actual_operands_verified')
    for foreach in (None,True):
        check();case_start=time.monotonic()
        # A real ParameterList preserves the exact recorded order and operands.
        # Its numeric names are an optimizer-only ownership shell,not the DiT.
        model_a=torch.nn.ParameterList([torch.nn.Parameter(initial[n].clone()) for n in names])
        opt_a=torch.optim.AdamW(model_a.parameters(),lr=1e-4,weight_decay=.01,foreach=foreach)
        progress_a=common.initial_optimizer_progress(model_a,opt_a)
        def update(model,opt,participation):
            nonlocal actual_updates
            check();opt.zero_grad(set_to_none=True)
            for p,name in zip(model,names):p.grad=gradient[name].clone()
            opt.step();actual_updates+=1
            return common.advance_optimizer_progress(participation,model,opt)
        for step in range(1,21):progress_a=update(model_a,opt_a,progress_a)
        saved=common.capture_model_state(model_a,opt_a,progress_a)
        common.validate_model_state(saved,model_a,opt_a,20)
        model_b=torch.nn.ParameterList([torch.nn.Parameter(initial[n].clone()) for n in names])
        opt_b=torch.optim.AdamW(model_b.parameters(),lr=1e-4,weight_decay=.01,foreach=foreach)
        common.validate_model_state(saved,model_b,opt_b,20)
        common._copy_model_state(saved,model_b,opt_b);progress_b=dict(saved['optimizer_progress'])
        restored=common.capture_model_state(model_b,opt_b,progress_b)
        before_equal=common.tree_sha256(saved)==common.tree_sha256(restored)
        if not before_equal:raise ValueError('CPU restored boundary differs before update')
        del saved,restored
        rows=[]
        for step in range(21,26):
            progress_a=update(model_a,opt_a,progress_a);progress_b=update(model_b,opt_b,progress_b)
            equal=all(torch.equal(a,b) for a,b in zip(model_a,model_b))
            maximum=max(float((a.detach().float()-b.detach().float()).abs().max()) for a,b in zip(model_a,model_b))
            rows.append(dict(global_step=step,parameters_equal=equal,parameter_max_abs=maximum))
            report('controlled_optimizer_updates',foreach=foreach,current_global_step=step)
        state_a=common.capture_model_state(model_a,opt_a,progress_a);state_b=common.capture_model_state(model_b,opt_b,progress_b)
        complete_equal=common.tree_sha256(state_a)==common.tree_sha256(state_b)
        result=dict(foreach=foreach,state_before_first_resumed_update_equal=before_equal,updates=rows,
                    complete_final_state_equal=complete_equal,control_updates=25,restored_updates=5,
                    wall_seconds=time.monotonic()-case_start,passed=complete_equal and all(v['parameters_equal'] for v in rows))
        results.append(result);report('controlled_case_returned')
        del state_a,state_b,model_a,model_b,opt_a,opt_b
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only optimizer probe initialized GPU')
    result=dict(passed=all(r['passed'] for r in results),cases=results,actual_optimizer_steps=actual_updates,
                actual_input_tensors=476,actual_input_elements=9922560,source_parameter_names_SHA256=common.json_sha256(names),
                operand_pins={n:pins[n] for n in ('initial.pt','postclip.pt')},parameter_and_gradient_dtype='torch.bfloat16',
                gradient_schedule='The exact recorded firstpostclipgradient is repeated;not the native training gradient sequence.',
                restoration='Actual shared common._copy_model_state on realCPU ParameterList/AdamW values.',
                native_full_model_loaded=False,native_model_updates=0,GPU_cards=0,GPU_initialized=False,
                CUDA_optimizer_or_backward_cause_qualified=False,model_resume_acceptance=False,formal_performance=False,Model_GO=False,
                wall_seconds=time.monotonic()-started)
    with (output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binding',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--epoch',type=float,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();value=run(json.loads(Path(args.binding).read_bytes()),args.output,epoch=args.epoch,deadline=args.deadline)
    print(json.dumps(value))
    if not value['passed']:raise SystemExit(1)


if __name__=='__main__':main()
