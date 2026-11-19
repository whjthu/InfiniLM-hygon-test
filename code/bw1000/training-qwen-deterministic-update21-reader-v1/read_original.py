"""Read the original strict-determinism pair and audit its first resumed update."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
RUN = ROOT/'training-qwen-deterministic-math-recovery-v1'
TRACE_ROOTS = (RUN/'continuous/train/continuous-tail', RUN/'resumed/train/resume-witness')
CAP = 2**20
COLUMNS = 'JobIDRaw,State%40,ExitCode,ElapsedRaw,NodeList'
EXPECTED = {step: ('COMPLETED', '0:0', 'a01r4n16')
            for step in ('1087656.425', '1087656.426', '1087656.427')}
NAME = re.compile(r'^base_model\.model\.model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])\.default\.weight$')
VALID_SHA = re.compile(r'^[a-f0-9]{64}$')
STEPS = ('entry', 'forward', 'returned')
EVENTS = tuple(f'{event}-{i}' for i in range(1, 5) for event in ('entry', 'loss')) + (
    'preclip','postclip','preoptimizer','postoptimizer','update-complete'
)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path, expected=None):
    with path.open('rb') as f:
        raw = f.read(CAP+1)
    needed = hashlib.sha256(raw).hexdigest()
    need(0 < len(raw) <= CAP and (expected is None or needed == expected), 'original receipt changed or too large')
    return json.loads(raw), dict(bytes=len(raw), SHA256=needed)


def accounting():
    def command(*a):
        return subprocess.run(a, timeout=12, text=True, capture_output=True, check=True).stdout.splitlines()
    rows = [line.split('|') for line in command('sacct','-n','-P','-j','1087656','--format='+COLUMNS)]
    exact = [row for row in rows if row[0] in EXPECTED]
    need(len(exact) == len(EXPECTED) and all((r[1],r[2],r[4]) == EXPECTED[r[0]] for r in exact), 'original numbered accounting mismatch')
    live = command('squeue','-h','-s','-j','1087656','-o','%i')
    need(not set(EXPECTED).intersection(live), 'original step still visible')
    return exact


def original_scope():
    host, host_pin = read(RUN/'outcome.json', '71c9ac1cfd9f23fb9067a17b15421f7d4ec0173e30fa4f8f0285e27b1cf249e9')
    native, native_pin = read(RUN/'native-outcome.json', '0f117ed0928ec7d3aade150c21b4e8ec5b077a87d9dbdaafbf656e95ced2878e')
    fresh, fresh_pin = read(RUN/'continuous/native-outcome.json', '3f499409b39376e87e8506b5cb1c0c9c3f0c989f68935cfc1a6dfabb5efc4685')
    resume, resume_pin = read(RUN/'resumed/native-outcome.json', '440cf0fedbd74f3f44e40c24d7a0bb3340e1f0f80b0d9cfd8e55c7713150e71e')
    need(host['keeper_passed'] and native['normal_native_exit'] and fresh['normal_native_exit']
         and resume['normal_native_exit'], 'original native pair did not close')
    after = host['after']
    need(after['step']=='1087656.427' and after['kfd_holder_count']==0 and after['owned_processes_remaining']==[]
         and after['boot_id']==host['before']['boot_id'] and after['pci']==host['before']['pci'], 'original cleanup identity mismatch')
    return dict(controller=host_pin,pair=native_pin,continuous=fresh_pin,resumed=resume_pin)


def contexts(row):
    value = [row[key] for key in STEPS]
    for entry in value:
        need(set(entry)=={'ddp_sync','accelerator_sync'} and all(type(v) is bool for v in entry.values()), 'unknown actual sync context')
    return value


def inventory(value):
    need(len(value)==504, 'expected all 504 gradients')
    counts = Counter()
    out = {}
    for name, item in value.items():
        match = NAME.fullmatch(name)
        need(match is not None, 'unexpected native trainable gradient name')
        layer, group, projection, factor = match.groups()
        layer = int(layer)
        need(0<=layer<36 and (group=='self_attn')==(projection in ('q_proj','k_proj','v_proj','o_proj')), 'wrong layer or projection group')
        counts[layer] += 1
        if item is None:
            out[name] = (layer, projection, factor, None, None)
        else:
            need(item['dtype']=='torch.float32' and VALID_SHA.fullmatch(item['SHA256']) and len(item['shape'])==2, 'wrong gradient dtype/shape/SHA')
            out[name] = (layer, projection, factor, item['shape'], item['SHA256'])
    need(len(counts)==36 and set(counts.values())=={14}, 'incomplete layer gradient inventory')
    return out


def compare_gradients(left, right):
    left, right = inventory(left), inventory(right)
    need(left.keys()==right.keys(), 'named input gradients differ')
    changed, projections, factors = Counter(), Counter(), Counter()
    presence_changes = 0
    for name, lrow in left.items():
        rrow = right[name]
        if lrow[3] is not None and rrow[3] is not None:
            need(lrow[3]==rrow[3], 'native gradient shape differs')
        presence_changes += (lrow[4] is None) != (rrow[4] is None)
        if lrow[4]!=rrow[4]:
            changed[lrow[0]] += 1
            projections[lrow[1]] += 1
            factors[lrow[2]] += 1
    return dict(changed_gradient_tensors=sum(changed.values()), gradient_presence_changes=presence_changes,
                by_layer=dict(sorted(changed.items())), by_projection=dict(sorted(projections.items())),
                by_factor=dict(sorted(factors.items())))


def rank_evidence(rank):
    docs, receipt_pins = [], []
    for root in TRACE_ROOTS:
        direct = root/f'rank-{rank:02d}'
        readset = [read(direct/n) for n in ('backward-21.json','update-21.json','result.json')]
        docs.append([x[0] for x in readset])
        receipt_pins.append([x[1] for x in readset])
    (left_b, left_u, left_r), (right_b, right_u, right_r) = docs
    for b,u,r in docs:
        need(b['complete'] and u['complete'] and r['rank']==rank and r['world_size']==8 and r['completed_global_step']==25,
             'incomplete original rank records')
        need([x['ordinal'] for x in b['microbatches']]==list(range(1,5)) and tuple(x['boundary'] for x in u['events'])==EVENTS,
             'first restored accumulation boundaries changed')
        need(len(r['inputs'])==len(r['outputs'])==20 and [x['global_step'] for x in r['inputs'][:4]]==[20]*4, 'original rank input ordinals differ')
    aev, bev = ({x['boundary']:x['value'] for x in t['events']} for t in (left_u,right_u))
    samples = []
    final_gradient_fingerprints = None
    for i, (a,b) in enumerate(zip(left_b['microbatches'],right_b['microbatches'],strict=True)):
        ac,bc = contexts(a),contexts(b)
        no_reduction = all(not flags['ddp_sync'] and not flags['accelerator_sync'] for flags in ac+bc)
        before = left_r['inputs'][i]['full_input']
        after = right_r['inputs'][i]['full_input']
        need(before['finite'] and after['finite'] and VALID_SHA.fullmatch(before['SHA256']) and VALID_SHA.fullmatch(after['SHA256']),
             'invalid original input evidence')
        ordinal = i+1
        loss_a = left_r['outputs'][i]['loss']
        loss_b = right_r['outputs'][i]['loss']
        need(loss_a==aev[f'loss-{ordinal}'] and loss_b==bev[f'loss-{ordinal}'], 'loss record differs across producer observers')
        change = compare_gradients(a['gradients'],b['gradients'])
        samples.append(dict(microbatch=ordinal,sync_context=ac,resumed_sync_context=bc,context_equal=ac==bc,recorded_no_reduction=no_reduction,
            input_equal=before['SHA256']==after['SHA256'], entry_rng_equal=aev[f'entry-{ordinal}']==bev[f'entry-{ordinal}'],
            loss_equal=loss_a['SHA256']==loss_b['SHA256'], **change))
        if i==3:
            final_gradient_fingerprints=[{k:v[4] for k,v in inventory(row['gradients']).items()} for row in (a,b)]
    def state(name):
        p,q = aev[name],bev[name]
        need(set(p)==set(q)=={'adapter','optimizer','scheduler'}, 'unexpected optimizer observation fields')
        return {key:p[key]==q[key] for key in p}
    first = next((s['microbatch'] for s in samples if s['changed_gradient_tensors'] or s['gradient_presence_changes']),None)
    loads = right_r['RNG_loads']
    need(len(loads)==1 and loads[0]['global_step']==20 and type(loads[0]['saved_equal']) is bool,
         'unexpected original checkpoint RNG restore witness')
    policy = []
    for phase in ('continuous','resumed'):
        record, pin = read(RUN/phase/'train'/f'math-sdpa-rank-{rank:02d}.json')
        need(record['phase']==phase and record['rank']==rank and record['completed_step']==25
             and record['mode']=='MATH' and record['deterministic'] is True and record['warn_only'] is False,
             'actual provider policy differs')
        policy.append(pin)
    return dict(rank=rank, receipts=receipt_pins, policy_receipts=policy, microbatches=samples,
                saved_rng_restore_equal=loads[0]['saved_equal'],first_gradient_difference=first,
                preclip=compare_gradients(aev['preclip'],bev['preclip']),
                postclip=compare_gradients(aev['postclip']['gradients'],bev['postclip']['gradients']),
                clipping_norm_equal=aev['postclip']['returned_norm']==bev['postclip']['returned_norm'],
                preoptimizer_equal=state('preoptimizer'), postoptimizer_equal=state('postoptimizer')), final_gradient_fingerprints


def main():
    rows=accounting()
    scope=original_scope()
    pairs=[rank_evidence(rank) for rank in range(8)]
    ranks=[item[0] for item in pairs]
    local=[r['rank'] for r in ranks if any(x['recorded_no_reduction'] and x['changed_gradient_tensors'] for x in r['microbatches'])]
    same_before_clip=[r['rank'] for r in ranks if r['preclip']['changed_gradient_tensors']==0]
    cross_rank_fourth={name:all(p[1][i]==pairs[0][1][i] for p in pairs) for i,name in enumerate(('continuous','resumed'))}
    affected = [r for r in ranks if not r['microbatches'][0]['loss_equal']]
    need([row['rank'] for row in affected]==[2,6]
         and all(s['input_equal'] for r in ranks for s in r['microbatches']), 'first loss boundary does not reproduce prior qualification')
    return dict(first_microbatch_loss_diff=[row['rank'] for row in affected],
                first_microbatch_rng_diff=[r['rank'] for r in ranks if not r['microbatches'][0]['entry_rng_equal']],
                saved_rng_restore_mismatch=[r['rank'] for r in ranks if not r['saved_rng_restore_equal']],
                at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting_columns=COLUMNS.split(','),original_accounting=rows,
                actual_scope_pins=scope,planned_ranks=8,complete_ranks=len(ranks),planned_microbatches=32,
                complete_microbatches=sum(len(r['microbatches']) for r in ranks),
                local_gradient_drift_ranks=local,identical_cross_rank_fourth_within_each_branch=cross_rank_fourth,
                identical_preclip_ranks=same_before_clip,all_preoptimizer_state_equal=all(all(r['preoptimizer_equal'].values()) for r in ranks),
                ranks=ranks,started_new_steps=0,raw_tensor_export_bytes=0)


if __name__=='__main__':
    print(json.dumps(main(),allow_nan=False,separators=(',',':')))
