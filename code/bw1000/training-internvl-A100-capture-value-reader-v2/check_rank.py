"""One rank, CPU-only, exact original artifacts; never invoked during preparation."""
import argparse
import gc
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import time
import zipfile

D = Path(__file__).resolve().parent
R = Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1')
P = json.loads((D/'ORIGINAL_PINS.json').read_bytes())
B = json.loads((D/'INTERNVL_NATIVE_BINDING.json').read_bytes())
torch = None
deadline = None
CHUNK = 1024 * 1024
progress = {}


def check_time():
    if time.monotonic() >= deadline:
        raise TimeoutError('original reader work epoch exhausted')


def stable(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def original_json(rel):
    pin = P['originals'][rel]
    p = R/rel
    assert p.is_file() and not p.is_symlink() and p.stat().st_size == pin['bytes']
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pin['SHA256']
    return json.loads(raw)


def load_verified(path, pin):
    check_time()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode), 'original artifact must be a regular file'
        assert stable(before) == stable(path.stat()) and before.st_size == pin['bytes']
        assert 0 < before.st_size <= P['largest_artifact_bytes'] <= 2*1024**3-1
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4*CHUNK), b''):
            check_time()
            digest.update(block)
        assert digest.hexdigest() == pin['SHA256'], 'artifact SHA differs'
        assert stable(before) == stable(os.fstat(stream.fileno()))
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            assert len(entries) <= 30000
            assert all(i.compress_type == zipfile.ZIP_STORED for i in entries)
            assert sum(i.file_size for i in entries) <= before.st_size
            assert all(i.file_size <= 16*1024**2 for i in entries if i.filename.endswith('/data.pkl'))
        stream.seek(0)
        # Same verified open file description, no pathname reopen or unsafe fallback.
        value = torch.load(stream, weights_only=True, map_location='cpu')
        assert stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat())
    check_time()
    return value


def chunks(tensor):
    assert tensor.device.type == 'cpu' and tensor.layout == torch.strided
    assert tensor.is_contiguous(), 'unsupported noncontiguous artifact; do not copy silently'
    flat = tensor.detach().reshape(-1)
    for offset in range(0, flat.numel(), CHUNK):
        check_time()
        yield flat[offset:offset+CHUNK]


def tensor_hash(tensor):
    h = hashlib.sha256()
    for part in chunks(tensor):
        h.update(part.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def values(value, inventory, path='$'):
    if isinstance(value, torch.Tensor):
        for part in chunks(value):
            if part.is_floating_point() or part.is_complex():
                assert bool(torch.isfinite(part).all()), 'independent tensor finite failure'
        inventory.append(dict(path=path, shape=list(value.shape), dtype=str(value.dtype), numel=value.numel()))
    elif isinstance(value, dict):
        assert all(isinstance(k, (str,int)) for k in value)
        for k,v in value.items():
            values(v, inventory, path+'/'+str(k))
    elif isinstance(value, (list,tuple)):
        for i,v in enumerate(value):
            values(v, inventory, path+'/'+str(i))
    elif isinstance(value, float):
        assert math.isfinite(value), 'independent scalar finite failure'
    else:
        assert value is None or isinstance(value, (str,int,bool)), 'unsupported captured value type'


def parameters(value, manifest):
    assert set(value) == set(manifest)
    hashes = {}
    for name,tensor in value.items():
        m = manifest[name]
        assert list(tensor.shape) == m['shape'] and str(tensor.dtype) == m['dtype'] and tensor.numel() == m['numel']
        hashes[name] = tensor_hash(tensor)
    return hashes


def norm(gradients):
    squared = 0.0
    for value in gradients.values():
        if value is not None:
            for part in chunks(value):
                squared += float(torch.sum(part.to(dtype=torch.float64).square()))
    assert math.isfinite(squared)
    return math.sqrt(squared)


def gradients(value, manifest):
    assert set(value) == set(manifest)
    for name,tensor in value.items():
        if tensor is not None:
            m = manifest[name]
            assert list(tensor.shape) == m['shape'] and str(tensor.dtype) == m['dtype']
    return {n for n,t in value.items() if t is not None}


def optimizer(value, manifest, expected_step, active):
    assert set(value) == {'state_by_name','param_groups'}
    states,groups = value['state_by_name'],value['param_groups']
    assert set(states) == set(manifest)
    names = [n for group in groups for n in group['parameter_names']]
    assert len(names) == len(set(names)) == len(manifest) and set(names) == set(manifest)
    recipe = B['spec']['training']
    for group in groups:
        assert tuple(group['betas']) == (recipe['adam_beta1'],recipe['adam_beta2'])
        assert group['eps'] == recipe['adam_epsilon'] and group['weight_decay'] in (0.0,recipe['weight_decay'])
        assert 0 <= group['lr'] <= recipe['learning_rate']
        assert not group.get('amsgrad',False) and not group.get('maximize',False)
    initialized = set()
    for name,state in states.items():
        if not state:
            continue
        assert set(state) == {'step','exp_avg','exp_avg_sq'}
        step = state['step'].item() if isinstance(state['step'],torch.Tensor) else state['step']
        assert 0 <= step <= expected_step
        if name in active:
            assert step == expected_step
        for key in ('exp_avg','exp_avg_sq'):
            assert list(state[key].shape) == manifest[name]['shape'] and str(state[key].dtype) == manifest[name]['dtype']
        initialized.add(name)
    if expected_step == 0:
        assert not initialized
    else:
        assert active <= initialized, 'active gradient lacks optimizer state'
    state_hash = hashlib.sha256()
    for name,state in sorted(states.items()):
        state_hash.update(name.encode())
        for key,item in sorted(state.items()):
            state_hash.update(key.encode())
            state_hash.update((tensor_hash(item) if isinstance(item,torch.Tensor) else repr(item)).encode())
    return dict(initialized=len(initialized),empty=len(states)-len(initialized),step=expected_step,
                state_values_SHA256=state_hash.hexdigest(),
                group_count=len(groups),learning_rates=[g['lr'] for g in groups])


def main(rank, work_end):
    global torch,deadline,progress
    deadline = work_end
    assert all(os.environ.get(k) == '-1' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
    import torch as torch_module
    torch = torch_module
    # Distribution metadata canonicalizes nv25.06 to nv25.6; pin both identities.
    from importlib.metadata import version
    assert version('torch') == '2.8.0a0+5228986c39.nv25.6'
    assert str(torch.__version__) == '2.8.0a0+5228986c39.nv25.06'
    version_source = Path(torch.__file__).resolve().parent/'version.py'
    assert version_source.is_file() and not version_source.is_symlink()
    assert hashlib.sha256(version_source.read_bytes()).hexdigest() == 'f705f2955ebf304eab692c52e648caa82cfb45ae849979f74bc7ffde19706bd2'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    check_time()
    native = original_json('train/nvidia-baseline/rank-%02d.json'%rank)
    receipt = original_json('capture/rank-%02d/receipt.json'%rank)
    complete = original_json('capture/rank-%02d/native25-complete.json'%rank)
    assert complete == dict(rank=rank,global_step=25)
    assert native['rank'] == rank and native['world_size'] == 8 and native['global_step'] == 25
    assert [v['step'] for v in native['step_records']] == list(range(1,26))
    assert receipt['rank'] == rank and receipt['world_size'] == 8 and receipt['completed_global_step'] == 2
    assert (receipt['optimizer_updates'],receipt['micro_forwards'],receipt['clip_calls']) == (2,8,2)
    files = {f['name']:f for f in receipt['files']}
    assert len(files) == len(receipt['files']) == 25
    assert sum(f['bytes'] for f in files.values()) == P['capture_bytes_per_rank'][rank] <= 8*1024**3
    manifest = {m['name']:m for m in B['trainable_manifest']}
    result = dict(rank=rank,passed=False,files_verified=0,files_denominator=25,updates_denominator=2,
                  micro_forwards_denominator=8,independent_tensor_finite=True,files=[],updates=[])
    progress = result
    def take(name):
        value = load_verified(R/('capture/rank-%02d'%rank)/name,files[name])
        inventory = []
        values(value,inventory)
        raw = json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()
        result['files'].append(dict(name=name,bytes=files[name]['bytes'],SHA256=files[name]['SHA256'],
            tensor_count=len(inventory),tensor_elements=sum(v['numel'] for v in inventory),
            shape_dtype_inventory_SHA256=hashlib.sha256(raw).hexdigest(),dtypes=sorted({v['dtype'] for v in inventory}),
            independent_values_finite=True,stable_same_fd_SHA_load=True))
        result['files_verified'] += 1
        return value
    initial = take('initial.pt')
    assert initial['global_step'] == 0 and initial['manifest'] == B['trainable_manifest']
    previous = parameters(initial['parameters'],manifest)
    h = hashlib.sha256()
    # Exact d731 callback algorithm: names sorted, then tuple(shape), then bytes.
    for name,tensor in sorted(initial['parameters'].items()):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode())
        for part in chunks(tensor):h.update(part.view(torch.uint8).numpy().tobytes())
    assert h.hexdigest() == native['initial_trainable_sha256']
    result['initial_parameter_SHA_matches_native_rank'] = True
    del initial
    gc.collect()
    for update in (1,2):
        for ordinal in range((update-1)*4+1,update*4+1):
            for kind in ('input','output'):
                value = take('forward-%02d-%s.pt'%(ordinal,kind))
                assert value['global_step'] == update-1 and value['micro_ordinal'] == ordinal
                if kind == 'input':
                    kwargs = value['kwargs']
                    assert isinstance(value['args'],(list,tuple)) and isinstance(kwargs,dict)
                    if 'input_ids' in kwargs:
                        ids = kwargs['input_ids'];assert ids.dtype == torch.int64 and ids.ndim == 2 and ids.shape[0] == 1 and 0 < ids.shape[-1] <= 4096
                        if 'labels' in kwargs:assert kwargs['labels'].shape == ids.shape and kwargs['labels'].dtype == torch.int64
                else:
                    output = value['result'];assert isinstance(output,dict)
                    assert 'loss' in output and isinstance(output['loss'],torch.Tensor) and output['loss'].numel() == 1
                    if output.get('logits') is not None:assert output['logits'].shape[-1] == 151936
                del value
                gc.collect()
        pre = take('update-%02d-preclip.pt'%update)
        assert pre['global_step'] == update-1 and pre['max_grad_norm'] == B['spec']['training']['max_grad_norm']
        active = gradients(pre['gradients'],manifest)
        pre_norm = norm(pre['gradients'])
        post = take('update-%02d-postclip.pt'%update)
        assert post['global_step'] == update-1 and gradients(post['gradients'],manifest) == active
        post_norm = norm(post['gradients'])
        reported = float(post['reported_preclip_norm'])
        assert math.isfinite(reported) and reported >= 0
        # Record the independent norms. No unqualified tolerance creates clipping parity.
        row = dict(update=update,gradient_parameters=len(active),none_gradients=len(manifest)-len(active),
                   independent_preclip_L2=pre_norm,independent_postclip_L2=post_norm,
                   original_reported_preclip_norm=reported,clip_numerical_parity_qualified=False)
        del pre,post
        gc.collect()
        before = take('update-%02d-before-optimizer.pt'%update)
        assert before['global_step'] == update-1
        before_hashes = parameters(before['parameters'],manifest)
        assert before_hashes == previous, 'parameter continuity differs at native optimizer boundary'
        row['optimizer_before'] = optimizer(before['optimizer'],manifest,update-1,set())
        if update == 2:
            assert row['optimizer_before']['state_values_SHA256'] == result['updates'][0]['optimizer_after']['state_values_SHA256']
        del before
        gc.collect()
        after = take('update-%02d-after-optimizer.pt'%update)
        assert after['global_step_before_increment'] == update-1
        previous = parameters(after['parameters'],manifest)
        row['optimizer_after'] = optimizer(after['optimizer'],manifest,update,active)
        row['parameters_changed'] = sum(previous[n] != before_hashes[n] for n in previous)
        row['parameters_denominator'] = len(manifest)
        result['updates'].append(row)
        del after
        gc.collect()
    assert result['files_verified'] == 25 and len(result['updates']) == 2
    assert not torch.cuda.is_initialized()
    result.update(passed=True,capture_bytes=sum(f['bytes'] for f in files.values()),actual_GPU_cards=0,
                  torch_version=str(torch.__version__),cpu_only=True,full_optimizer_recompute=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--work-end',type=float,required=True)
    args = parser.parse_args()
    assert 0 <= args.rank < 8
    try:
        value = main(args.rank,args.work_end)
    except Exception as error:
        import traceback
        value = dict(progress,rank=args.rank,passed=False,error_type=type(error).__name__,detail=str(error)[:250],
                     failure_line=traceback.extract_tb(error.__traceback__)[-1].lineno,files_denominator=25,updates_denominator=2,
                     independent_tensor_finite=False,actual_GPU_cards=0)
    print(json.dumps(value,allow_nan=False))
    raise SystemExit(0 if value['passed'] else 1)
