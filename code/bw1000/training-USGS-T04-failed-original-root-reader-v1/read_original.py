"""Read two saved original JSON bundles; never import a training framework."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

REQUEST = '889e2c96be9d4546bee228f6ec527bc0'
PAIR = 'd58ebcf4dc0345b2a3719067f0fe6961'
TERMINAL = '36aa86ffaf2d402bd31ba3132da81c66c6c8b18408e79facf813a1d0d4403c61'
LEDGER = '7e34e226ef08d879f4f1ff6c982ead01495e850e95be97fd3ab94b679b8877dc'


def need(value, reason):
    if not value:
        raise ValueError(reason)


def load(path, expected, cap):
    raw = Path(path).read_bytes()
    need(len(raw) <= cap and hashlib.sha256(raw).hexdigest() == expected,
         'ORIGINAL_BUNDLE_PIN')
    return json.loads(raw)


def audit(terminal, ledger):
    files = terminal['files']
    outcome = files['run/outcome.json']['value']
    need(outcome['request_id'] == REQUEST and outcome['pair_id'] == PAIR
         and outcome['native_returncode'] == -15
         and outcome['engineering_completed'] is False, 'ORIGINAL_FAILURE')
    need(files['run/failure-private.json']['value']['message'] == 'OWNED_VMRSS_SCHEMA',
         'ORIGINAL_FAILURE_REASON')
    need(outcome['cleanup']['owned_processes_remaining'] == 0
         and outcome['cleanup']['held_child']['reaped'] is True
         and outcome['release_permitted'] is True, 'ORIGINAL_CLEANUP')
    post = files['run/after.json']['value']
    need(post == outcome['post'] and post['active_compute_processes'] == 0
         and post['master_port_bindable'] is True and len(post['devices']) == 8
         and len({d['UUID'] for d in post['devices']}) == 8
         and len({d['PCI'] for d in post['devices']}) == 8, 'ORIGINAL_POST')
    need(ledger['request_id'] == REQUEST and ledger['GPU_calls'] == 0
         and ledger['remote_writes'] == 0, 'INDEPENDENT_LEDGER_SCOPE')
    ranks, counts, tokens, supervised = [], collections.Counter(), 0, 0
    adapter, scheduler, initial = set(), set(), set()
    for rank in range(8):
        prefix = 'run/witness/rank-%02d/' % rank
        owner = files[prefix + 'owner.json']['value']
        need(owner['request_id'] == REQUEST and owner['pair_id'] == PAIR
             and owner['rank'] == rank, 'ACTUAL_RANK_OWNER')
        progress = files[prefix + 'progress.json']['value']
        need(progress['completed_microbatches'] == 100
             and progress['completed_updates'] == list(range(1, 26))
             and progress['saved_steps'] == [10, 20, 25]
             and progress['error'] is None, 'ACTUAL_RANK_UPDATES')
        receipt = files[prefix + 'receipt.json']
        if rank == 0:
            need(receipt == {'exists': False} and progress['finished'] is False,
                 'ORIGINAL_MISSING_RANK0_MUST_REMAIN')
        else:
            value = receipt['value']
            need(progress['finished'] is True and value['rank'] == rank
                 and value['pair_id'] == PAIR and value['updates'] == 25
                 and value['completed_microbatches'] == 100
                 and value['completed_step'] == 25
                 and receipt['SHA256'] == outcome['rank_receipts'][str(rank)]['file']['SHA256'],
                 'ORIGINAL_RANK_RECEIPT')
        first = files[prefix + 'initial-state.json']['value']
        need(first['tensors'] == 504 and first['elements'] == 43646976
             and first['global_step'] == 0 and first['RNG_restored'] is False,
             'ACTUAL_INITIAL')
        initial.add(first['named_SHA256'])
        key = 'witness/rank-%02d/' % rank
        rows = ledger['files'][key + 'microbatches.json']['value']
        inputs, forwards, losses = (rows[k] for k in ('training_inputs', 'forward_inputs', 'losses'))
        need(rows['pair_id'] == PAIR and rows['mode'] == 'fresh25'
             and rows['arm'] == 'domain' and len(inputs) == len(forwards) == len(losses) == 100
             and all(type(x) in (int, float) and math.isfinite(x) for x in losses),
             'ACTUAL_COMPLETE_MICROBATCHES')
        local = collections.Counter()
        for i, (row, forward) in enumerate(zip(inputs, forwards)):
            need(row['microbatch'] == forward['microbatch'] == i
                 and row['global_step_before'] == i // 4
                 and type(row['canonical_ordinal']) is int and 0 <= row['canonical_ordinal'] < 45
                 and row['input_tokens'] > 0 and row['effective_supervised_tokens'] > 0
                 and row['supervised_EOS_count'] > 0
                 and forward['RNG_before']['finite'] is True,
                 'ACTUAL_MICROBATCH_SEQUENCE')
            local[row['canonical_ordinal']] += 1
        need({str(k): v for k, v in local.items()} == rows['canonical_occurrences'],
             'ACTUAL_LOCAL_OCCURRENCES')
        it = sum(r['input_tokens'] for r in inputs)
        st = sum(r['effective_supervised_tokens'] for r in inputs)
        need(it == rows['actual_input_tokens'] and st == rows['actual_supervised_tokens'],
             'ACTUAL_TOKEN_DENOMINATORS')
        counts.update(local)
        tokens += it
        supervised += st
        after = ledger['files'][key + 'update-25-after.json']['value']
        sched = ledger['files'][key + 'update-25-scheduler.json']['value']
        need(after['optimizer_steps'] == sched['global_step'] == 25
             and sched['epoch'] == 1.0, 'ACTUAL_FINAL_UPDATE')
        adapter.add(after['adapter_SHA256'])
        scheduler.add(sched['scheduler_SHA256'])
        ranks.append(dict(rank=rank, microbatches=100, updates=25,
                          receipt_present=rank != 0, finished=progress['finished'],
                          input_tokens=it, supervised_tokens=st))
    actual = [counts[i] for i in range(45)]
    need(sum(actual) == 800 and min(actual) == 17 and max(actual) == 18
         and len(initial) == len(adapter) == len(scheduler) == 1, 'ALL8_AGREEMENT')
    for rank in range(8):
        global_counts = ledger['files']['witness/rank-%02d/global800-occurrences.json' % rank]['value']
        need(global_counts == dict(counts=actual, total=800), 'ACTUAL_ALLREDUCE_OCCURRENCES')
    checkpoints = {}
    for step in (10, 20, 25):
        pin = files['run/witness/rank-00/checkpoint-%d-files.json' % step]['value']
        need(len(pin['files']) == 14 and all(v['bytes'] > 0 for v in pin['files'].values()),
             'ORIGINAL_FULL_STATE_MANIFEST')
        checkpoints[str(step)] = dict(files=14, bytes=sum(v['bytes'] for v in pin['files'].values()),
                                      manifest_SHA256=files['run/witness/rank-00/checkpoint-%d-files.json' % step]['SHA256'])
    resource = outcome['resources']
    need(resource['soft_limit_exceeded'] is False, 'NO_RECORDED_RSS_LIMIT_EXCESS')
    return dict(request_id=REQUEST, pair_id=PAIR, original_state='FAIL',
                independent_metadata_consumption='PASS', native_returncode=-15,
                actual_GPU_training_started=True, ranks=ranks, actual_global_microbatches=800,
                actual_unique_train_rows=45, actual_input_tokens=tokens,
                actual_shifted_supervised_tokens=supervised, completed_rank_receipts=7,
                missing_rank_receipts=[0], live_adapter_all8_SHA256=next(iter(adapter)),
                live_scheduler_all8_SHA256=next(iter(scheduler)), initial_named_SHA256=next(iter(initial)),
                checkpoints=checkpoints, sampled_group_peak_RSS_bytes=resource['sampled_group_peak_RSS_bytes'],
                failure='OWNED_VMRSS_SCHEMA', memory_limit_exceeded=False,
                exit_race_cause='HYPOTHESIS_NO_OFFENDING_STATUS_SAVED', clean_post=True,
                termination_needed=True, release_permitted=True, replay_allowed=False,
                saved25_equals_live_optimizer='UNPROVEN_FULL_LIVE_DIGEST_NOT_SAVED',
                checkpoint_restore_qualified=False, domain_model_qualified=False,
                formal17=False, performance_qualified=False, effect_qualified=False,
                new_training_calls=0, new_GPU_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(load(args.terminal, TERMINAL, 65536), load(args.ledger, LEDGER, 2 * 1024**2))
    with Path(args.output).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(original_state=result['original_state'],
                          actual_global_microbatches=result['actual_global_microbatches'],
                          completed_rank_receipts=result['completed_rank_receipts'])))


if __name__ == '__main__':
    main()
