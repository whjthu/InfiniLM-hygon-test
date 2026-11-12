"""Fresh-process CUDA optimizer diagnosis with actual recorded BF16 operands.

Repeat the original first-step postclip gradient as a controlled trajectory.
This is not the actual model gradient sequence, model training or performance.
Control and resume are separate Python processes under one external GPU owner.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))


def run(binding,output,*,arm,foreach,epoch,deadline):
    import torch
    from gencuda import hy_resume_checkpoint as common
    if not all(math.isfinite(v) for v in (epoch,deadline)) or not 0<deadline-epoch<=600:
        raise ValueError('invalid original CUDA epoch')
    anchor=time.monotonic();remaining=deadline-time.time()
    def check():
        if min(deadline-time.time(),remaining-(time.monotonic()-anchor))<=0:
            raise TimeoutError('CUDA optimizer original epoch expired')
    def sha(raw):return hashlib.sha256(raw).hexdigest()
    def read_pin(p,pin):
        check()
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=pin['bytes']:
            raise ValueError('original recorded operand extent differs')
        raw=p.read_bytes()
        if sha(raw)!=pin['SHA256']:raise ValueError('original recorded operand bytes differ')
        return raw
    root=Path(binding['reference_root']);mp=root/binding['capture_manifest']['relative_path']
    manifest=json.loads(read_pin(mp,binding['capture_manifest']))
    if manifest['phase']!='NATIVE25_RETURNED' or manifest['completed_global_steps']!=25 or not manifest['first_update_complete']:
        raise ValueError('original native25 first-update capture incomplete')
    pins={p['name']:p for p in manifest['files']}
    initial=torch.load(io.BytesIO(read_pin(mp.parent/'initial.pt',pins['initial.pt'])),map_location='cpu',weights_only=True)
    gradient=torch.load(io.BytesIO(read_pin(mp.parent/'postclip.pt',pins['postclip.pt'])),map_location='cpu',weights_only=True)
    names=list(initial)
    if len(names)!=476 or set(names)!=set(gradient) or sum(v.numel() for v in initial.values())!=9922560:
        raise ValueError('full actual recorded parameter set differs')
    for name,v in initial.items():
        g=gradient[name]
        if v.device.type!='cpu' or v.dtype!=torch.bfloat16 or not v.is_contiguous() or g.device.type!='cpu' or g.dtype!=v.dtype or g.shape!=v.shape or not g.is_contiguous():
            raise ValueError('recorded CPU BF16 operand contract differs')
        if not bool(torch.isfinite(v).all()) or not bool(torch.isfinite(g).all()):raise ValueError('nonfinite actual operand')
    if torch.cuda.device_count()!=1:raise ValueError('exactly one visible CUDA device required')
    torch.set_num_threads(1);torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info();floor=16*1024**3
    if free<floor:raise ValueError('strict 16GiB CUDA floor')
    torch.cuda.reset_peak_memory_stats()
    output=Path(output);case=output/('default' if foreach is None else 'foreach-off')
    if arm=='control':case.mkdir(mode=0o700,parents=True,exist_ok=False)
    elif arm!='resume' or not case.is_dir():raise ValueError('unknown arm or missing original control')
    def dump(name,value):
        with (case/name).open('x') as f:json.dump(value,f,indent=2)
    def record(phase,step):
        check();torch.cuda.synchronize();current_free,_=torch.cuda.mem_get_info()
        if current_free<floor:raise ValueError('strict16GiB floor crossed')
        p=case/(arm+'-progress.tmp');p.write_text(json.dumps(dict(phase=phase,completed_global_step=step,arm=arm,foreach=foreach,epoch=epoch,deadline=deadline)))
        p.replace(case/(arm+'-progress.json'))
    model=torch.nn.ParameterList([torch.nn.Parameter(initial[n].to('cuda')) for n in names])
    gradients=[gradient[n].to('cuda') for n in names]
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,foreach=foreach)
    progress=common.initial_optimizer_progress(model,opt);completed=0;before_equal=None;rows=[]
    def update():
        nonlocal progress,completed
        check();opt.zero_grad(set_to_none=True)
        for p,g in zip(model,gradients):p.grad=g.clone()
        opt.step();torch.cuda.synchronize();completed+=1
        progress=common.advance_optimizer_progress(progress,model,opt)
    def save_state(step):
        state=common.capture_model_state(model,opt,progress)
        path=case/('state-%d.private.pt'%step)
        with path.open('xb') as f:torch.save(state,f)
        if sum(p.stat().st_size for p in output.glob('*/*.private.pt'))>1024**3:
            raise ValueError('private checkpoint budget exceeded')
        return state,dict(bytes=path.stat().st_size,SHA256=common.file_sha256(path),state_SHA256=common.tree_sha256(state))
    if arm=='resume':
        control=json.loads((case/'control-result.json').read_bytes());pin=control['checkpoints']['20'];cp=case/'state-20.private.pt'
        saved=torch.load(io.BytesIO(read_pin(cp,pin)),map_location='cpu',weights_only=True)
        common.validate_model_state(saved,model,opt,20)
        common._copy_model_state(saved,model,opt);progress=dict(saved['optimizer_progress'])
        before_equal=common.tree_sha256(saved)==common.tree_sha256(common.capture_model_state(model,opt,progress))
        if not before_equal:raise ValueError('restored full boundary differs before first resumed update')
        del saved
    checkpoints={};record('ready',0 if arm=='control' else 20)
    for step in range(1 if arm=='control' else 21,26):
        update();record('updated',step)
        if arm=='control' and step>=20:
            state,pin=save_state(step);checkpoints[str(step)]=pin;del state
        elif arm=='resume':
            state=common.capture_model_state(model,opt,progress)
            pin=control['checkpoints'][str(step)];cp=case/('state-%d.private.pt'%step)
            expected=torch.load(io.BytesIO(read_pin(cp,pin)),map_location='cpu',weights_only=True)
            equal=common.tree_sha256(state)==common.tree_sha256(expected)
            maximum=max(float((state['trainable'][n].float()-expected['trainable'][n].float()).abs().max()) for n in state['trainable'])
            rows.append(dict(global_step=step,complete_state_equal=equal,parameter_max_abs=maximum,
                component_equal={n:common.tree_sha256(state[n])==common.tree_sha256(expected[n]) for n in state}))
            del state,expected
    record('returned',25)
    result=dict(passed=arm=='control' or (before_equal and all(v['complete_state_equal'] for v in rows)),arm=arm,foreach=foreach,
        actual_optimizer_updates=completed,actual_operand_tensors=476,actual_operand_elements=9922560,source_parameter_names_SHA256=common.json_sha256(names),
        operand_pins={n:pins[n] for n in ('initial.pt','postclip.pt')},checkpoints=checkpoints,state_before_first_resumed_update_equal=before_equal,updates=rows,
        epoch=epoch,deadline=deadline,wall_seconds=time.monotonic()-anchor,torch_version=torch.__version__,CUDA_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0),device_total_bytes=total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_free_before_bytes=free,strict_free_floor_bytes=floor,gradient_schedule='Repeat the actual firstpostclipgradient;not native gradients20/21.',
        model_loaded=False,native_model_updates=0,formal_performance=False,native_resume_acceptance=False,Model_GO=False)
    dump(arm+'-result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=['control','resume'],required=True);p.add_argument('--foreach',choices=['default','off'],required=True)
    p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    v=run(json.loads(Path(a.binding).read_bytes()),a.output,arm=a.arm,foreach=None if a.foreach=='default' else False,epoch=a.epoch,deadline=a.deadline)
    print(json.dumps(v))
    if not v['passed']:raise SystemExit(1)


if __name__=='__main__':main()
