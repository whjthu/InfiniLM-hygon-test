"""Compare private HunyuanVideo first-update captures using CPU reductions."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from hygon_sdxl_compare_first_step import check_tensor, exact, require, vector_metrics


PHASE_FIELDS = {
    'before': {'params', 'optimizer', 'latents', 'condition', 'text', 'text_mask', 'byt5', 'byt5_mask'},
    'inputs': {'noise', 'timestep', 'hidden_states', 'target'},
    'forward': {'prediction', 'loss'},
    'preclip': {'gradients'},
    'postclip': {'gradients', 'norm'},
    'updated': {'params', 'optimizer'},
}
A100_CAPTURE_SHA256 = '81dae83599f86e0645a5a7478f5017f0d1ef7a8e028875839bc5f594f7cf08d5'


def load_capture(torch, root, *, reference):
    receipt_path = root / 'capture-result.json'
    with receipt_path.open('rb') as f:
        raw = f.read(16385)
    require(len(raw) <= 16384, 'oversized capture receipt')
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(raw)
    require(receipt['case'] == 'T09.sft' and type(receipt['actual_global_step']) is int
            and receipt['actual_global_step'] == 1 and receipt['performance_eligible'] is False,
            'unexpected HunyuanVideo capture boundary')
    if reference:
        require(digest == A100_CAPTURE_SHA256, 'original A100 capture receipt differs')
    else:
        require(receipt.get('fixed_a100_input_replay') is True, 'Hygon fixed-input replay evidence is missing')

    records = receipt['artifacts']
    require([row['phase'] for row in records] == list(PHASE_FIELDS), 'unexpected capture phase order')
    total = sum(row['bytes'] for row in records)
    require(total == receipt['artifact_bytes'] and 0 < total <= 256 * 1024**2,
            'unexpected capture byte total')
    phases = {}
    for row in records:
        phase = row['phase']
        name = phase + '.pt'
        size = row['bytes']
        require(row['file'] == name and type(size) is int and 0 < size <= 128 * 1024**2,
                'invalid capture artifact extent or name')
        with (root / name).open('rb') as f:
            raw = f.read(size + 1)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == row['SHA256'],
                f'{phase}: capture bytes or hash differ')
        value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        require(type(value) is dict and value.keys() == PHASE_FIELDS[phase], f'{phase}: field inventory differs')
        phases[phase] = value
    return phases, digest


def momentum(torch, phases):
    params = phases['before']['params']
    names = list(params)
    require(len(names) == 216 and sum(p.numel() for p in params.values()) == 1769472,
            'HunyuanVideo trainable inventory differs')
    before = phases['before']['optimizer']
    after = phases['updated']['optimizer']
    require(set(before) == set(after) == {'param_groups', 'state'}, 'unexpected Muon optimizer schema')
    groups = before['param_groups']
    require(len(groups) == 1 and groups[0]['params'] == list(range(len(names))),
            'Muon IDs do not map to trainable parameter order')
    exact(torch, after['param_groups'], groups, 'Muon parameter groups')
    require(set(before['state']) == set(after['state']) == set(groups[0]['params']),
            'Muon state inventory differs')
    tensors = {}
    for index, name in enumerate(names):
        exact(torch, before['state'][index], {'use_muon': True}, f'{name}/initial Muon state')
        state = after['state'][index]
        require(set(state) == {'use_muon', 'momentum_buffer'} and state['use_muon'] is True,
                f'{name}: unexpected updated Muon state')
        buffer = state['momentum_buffer']
        check_tensor(torch, buffer, params[name], f'{name}/momentum_buffer')
        tensors[name] = buffer
    return tensors


def compare(torch, observed, reference):
    exact(torch, observed['before'], reference['before'], 'initial LoRA, Muon and conditioning')
    exact(torch, observed['inputs'], reference['inputs'], 'fixed noise, timestep, hidden states and target')
    initial = reference['before']['params']
    metrics = {}

    def vectors(name, observed_values, reference_values, start=None):
        report = vector_metrics(torch, observed_values, reference_values, name, start)
        del report['per_tensor']
        del report['worst_relative_l2']
        metrics[name] = report

    vectors('prediction', {'prediction': observed['forward']['prediction']},
            {'prediction': reference['forward']['prediction']})
    for phase, name in (('preclip', 'gradient_before_clip'), ('postclip', 'gradient_after_clip')):
        require(list(reference[phase]['gradients']) == list(initial), f'{name}: trainable names differ')
        vectors(name, observed[phase]['gradients'], reference[phase]['gradients'])
    require(list(reference['updated']['params']) == list(initial), 'updated trainable names differ')
    vectors('parameter_update', observed['updated']['params'], reference['updated']['params'], initial)
    vectors('muon_momentum', momentum(torch, observed), momentum(torch, reference))
    for phase, field, name in (('forward', 'loss', 'loss'), ('postclip', 'norm', 'gradient_norm')):
        left, right = observed[phase][field], reference[phase][field]
        check_tensor(torch, left, right, name)
        require(right.is_floating_point() and right.numel() == 1, f'{name}: expected one floating value')
        a, b = float(left), float(right)
        metrics[name] = {'observed': a, 'reference': b, 'absolute_error': abs(a-b)}
    for name in ('gradient_before_clip', 'parameter_update', 'muon_momentum'):
        require(all(metrics[name][k] > 0 for k in ('reference_l2', 'observed_l2')), f'{name}: zero update evidence')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists')
    import torch

    reference, reference_hash = load_capture(torch, args.reference, reference=True)
    observed, observed_hash = load_capture(torch, args.observed, reference=False)
    result = dict(case='T09.sft', state='MEASURED_THRESHOLDS_PENDING', qualified=False,
                  reference_capture_SHA256=reference_hash, observed_capture_SHA256=observed_hash,
                  metrics=compare(torch, observed, reference))
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
