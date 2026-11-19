"""Attribute the SDXL first-step mismatch using the archived optimizer operands."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def load(path, pin):
    require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
            'ARCHIVE_SIZE_AND_HASH')
    return torch.load(path, map_location='cpu', weights_only=True)


def group(payload, count):
    saved = payload['initial_optimizer']
    result, = saved['param_groups']
    require(not saved['state'] and result['params'] == list(range(count)), 'FRESH_ADAMW_STATE')
    require(result['decoupled_weight_decay'] is True and not any(result.get(k, False) for k in
            ('amsgrad', 'maximize', 'capturable', 'differentiable')), 'ADAMW_FIRST_STEP_VARIANT')
    lr, wd, eps = (result[k] for k in ('lr', 'weight_decay', 'eps'))
    beta1, beta2 = result['betas']
    require(all(math.isfinite(x) for x in (lr, wd, eps, beta1, beta2))
            and lr > 0 and eps > 0 and wd >= 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1,
            'ADAMW_OPTIONS')
    return result


def checked_tensor(table, name, shape):
    value = table[name]
    require(value.device.type == 'cpu' and value.dtype == torch.bfloat16
            and value.shape == shape and bool(torch.isfinite(value).all()), 'TENSOR_SCHEMA')
    return value


def cpu_replay(initial, grads, opts):
    params = [torch.nn.Parameter(p.clone()) for p in initial]
    optim = torch.optim.AdamW(params, lr=opts['lr'], betas=opts['betas'], eps=opts['eps'],
                              weight_decay=opts['weight_decay'], foreach=False, fused=False)
    for p, grad in zip(params, grads):
        p.grad = grad.clone()
    optim.step()
    states = [optim.state[p] for p in params]
    require(all(int(s['step'].item()) == 1 for s in states), 'CPU_OPTIMIZER_STEP')
    return [p.detach() for p in params], [(s['exp_avg'], s['exp_avg_sq']) for s in states]


def analytic(initial, gradient, opts):
    # For an empty AdamW optimizer, bias correction makes the analytic step
    # proportional to g / (abs(g) + eps). Keep only the final BF16 rounding.
    p = initial.double()
    g = gradient.double()
    lr, eps = opts['lr'], opts['eps']
    result = p * (1 - lr * opts['weight_decay']) - lr * g / (g.abs() + eps)
    return result.to(initial.dtype)


def squared(tensor):
    return float(tensor.square().sum().item())


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(torch.version.hip and torch.cuda.device_count() == 0 and not torch.cuda.is_initialized(),
            'CPU_ONLY_ORIGINAL_HIP_IMAGE')
    cfg = json.loads(Path('/work/CLAIM.json').read_bytes())
    require(sha(Path('/reference/policy.json')) == cfg['policy_sha256'], 'ORIGINAL_POLICY')
    ref = load(Path('/reference/reference.pt'), cfg['archives']['A100'])
    hip = load(Path('/observed/observed.pt'), cfg['archives']['Hygon'])
    names = list(ref['initial_adapter'])
    count = len(names)
    keys = ('initial_adapter', 'final_adapter', 'gradients_before_clip', 'gradients_after_clip')
    require(count == 1120 and all(list(book[k]) == names for book in (ref, hip) for k in keys),
            'EXACT_ALL_1120_TRAINABLES')
    config = group(ref, count)
    require(group(hip, count) == config, 'IDENTICAL_ORIGINAL_ADAMW_OPTIONS')
    for book in (ref, hip):
        end = book['final_optimizer']
        group_end, = end['param_groups']
        require(all(group_end.get(k) == config[k] for k in ('lr','weight_decay','betas','eps'))
                and list(end['state']) == list(range(count)), 'ORIGINAL_FINAL_OPTIMIZER_SCHEMA')
    initial = [checked_tensor(ref['initial_adapter'], name, ref['initial_adapter'][name].shape)
               for name in names]
    for name, a in zip(names, initial):
        require(torch.equal(a, checked_tensor(hip['initial_adapter'], name, a.shape)), 'EXACT_INITIAL_VALUES')
    grads = [[checked_tensor(book['gradients_after_clip'], n, a.shape) for n,a in zip(names,initial)]
             for book in (ref, hip)]
    preclip = [[checked_tensor(book['gradients_before_clip'], n, a.shape) for n,a in zip(names,initial)]
               for book in (ref, hip)]
    require(all(torch.equal(pre, post) for past, present in zip(preclip, grads)
                for pre, post in zip(past, present)), 'NO_CLIP_CHANGE_IN_ARCHIVED_STEP')
    cpu, cpu_moments = zip(*(cpu_replay(initial, gg, config) for gg in grads))
    err2 = {key: 0.0 for key in ('native_cross', 'fixed_CPU_optimizer_cross', 'analytic_cross',
            'A100_same_grad_CPU', 'Hygon_same_grad_CPU', 'A100_same_grad_analytic','Hygon_same_grad_analytic',
            'A100_update', 'Hygon_update', 'CPU_A100_update', 'analytic_A100_update',
            'native_error_opposite_sign', 'native_error_exactly_one_zero', 'native_error_same_sign_nonzero',
            'native_error_both_zero', 'A100_same_grad_m', 'Hygon_same_grad_m', 'A100_same_grad_v',
            'Hygon_same_grad_v', 'A100_m', 'Hygon_m', 'A100_v', 'Hygon_v')}
    bins = {k: [0, 0, 0.0] for k in ('both_within_eps', 'up_to_16eps', 'above_16eps')}
    sign_counts = {k: 0 for k in ('opposite_sign', 'exactly_one_zero', 'same_sign_nonzero','both_zero')}
    exact_cpu_output_elements = [0,0]
    total = 0
    for idx, (n, init) in enumerate(zip(names, initial)):
        a, b = (g[idx] for g in grads)
        work = [checked_tensor(book['final_adapter'], n, init.shape) for book in (ref, hip)]
        p, q = (value.double() - init.double() for value in work)
        c, d = (rows[idx].double() - init.double() for rows in cpu)
        x,y = (analytic(init, grad, config).double() - init.double() for grad in (a,b))
        for key, value in (('native_cross',q-p),('fixed_CPU_optimizer_cross',d-c),('analytic_cross',y-x),
            ('A100_same_grad_CPU',c-p),('Hygon_same_grad_CPU',d-q),('A100_same_grad_analytic',x-p),
            ('Hygon_same_grad_analytic',y-q),('A100_update',p),('Hygon_update',q),
            ('CPU_A100_update',c),('analytic_A100_update',x)):
            err2[key] += squared(value)
        for side, tag in enumerate(('A100','Hygon')):
            saved = (ref, hip)[side]['final_optimizer']['state'][idx]
            require(int(saved['step'].item()) == 1, 'ORIGINAL_STEP_COUNTER')
            for key, short, actual in zip(('exp_avg','exp_avg_sq'),('m','v'),cpu_moments[side][idx]):
                expected = checked_tensor(saved,key,init.shape)
                err2[tag+'_same_grad_'+short] += squared(actual.double()-expected.double())
                err2[tag+'_'+short] += squared(expected.double())
            exact_cpu_output_elements[side] += int((cpu[side][idx] == work[side]).count_nonzero())
        both_zero = (a == 0) & (b == 0)
        one_zero = (a == 0) ^ (b == 0)
        opposite = (a.sign() == -b.sign()) & ~(both_zero | one_zero)
        same = ~(both_zero | one_zero | opposite)
        for label, mask in (('opposite_sign',opposite),('exactly_one_zero',one_zero),
                            ('same_sign_nonzero',same),('both_zero',both_zero)):
            sign_counts[label] += int(mask.count_nonzero())
            err2['native_error_'+label] += squared((q-p)[mask])
        magnitude = torch.maximum(a.double().abs(),b.double().abs())
        eps = config['eps']
        for label, mask in (('both_within_eps', magnitude<=eps),
                            ('up_to_16eps', (magnitude>eps)&(magnitude<=16*eps)),
                            ('above_16eps', magnitude>16*eps)):
            row = bins[label]
            row[0] += int(mask.count_nonzero())
            row[1] += int((mask & (p!=q)).count_nonzero())
            row[2] += squared((q-p)[mask])
        total += init.numel()
    require(sum(sign_counts.values())==total and sum(v[0] for v in bins.values())==total
            and math.isclose(sum(v[2] for v in bins.values()), err2['native_cross'], rel_tol=1e-12),
            'EXACT_POPULATION_AND_ERROR_PARTITION')
    require(math.isclose(math.sqrt(err2['native_cross']/err2['A100_update']),0.19271900631141353,rel_tol=1e-9),
            'ORIGINAL_PARAMETER_UPDATE_ERROR_REPRODUCED')
    denominators = dict(native_cross='A100_update',fixed_CPU_optimizer_cross='CPU_A100_update',
        analytic_cross='analytic_A100_update',A100_same_grad_CPU='A100_update',
        Hygon_same_grad_CPU='Hygon_update',A100_same_grad_analytic='A100_update',
        Hygon_same_grad_analytic='Hygon_update',A100_same_grad_m='A100_m',Hygon_same_grad_m='Hygon_m',
        A100_same_grad_v='A100_v',Hygon_same_grad_v='Hygon_v')
    result = dict(relative_l2={k: math.sqrt(err2[k]/err2[v]) for k,v in denominators.items()},
        native_error_squared=err2['native_cross'], exact_cpu_output_elements=exact_cpu_output_elements,
        sign_counts=sign_counts, native_error_squared_share_by_sign={k:err2['native_error_'+k]/err2['native_cross']
        for k in sign_counts}, archive_gradient_bins=bins,
        options={k:config.get(k) for k in ('lr','weight_decay','eps','betas','foreach','fused')},
        tensors=count,elements=total, GPU_initialized=torch.cuda.is_initialized(), physical_GPU=0,
        actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        torch=str(torch.__version__), hip=torch.version.hip,elapsed_seconds=time.monotonic()-started)
    with Path('/work/result.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
