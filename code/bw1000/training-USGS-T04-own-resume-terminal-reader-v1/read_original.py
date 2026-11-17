"""Offline saved-control metadata audit. Missing/failed evidence never becomes a PASS."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re

H = lambda raw: hashlib.sha256(raw).hexdigest()
PAIR = 'd58ebcf4dc0345b2a3719067f0fe6961'
INITIAL_SHA = 'b574c04a3bdaaff7f061d7285dc0fc73f2cbadce4801fca97dbd00ad8772fa0d'
NAMED_SHA = '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'
MEMBERS = {'adapter_config.json', 'adapter_model.safetensors', 'args.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json',
           *{'rng_state_%d.pth'%i for i in range(8)}}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def fingerprint(fields):
    return H(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode())


def canonical_PCI(value):
    need(isinstance(value, str), 'PCI_STRING')
    match = re.fullmatch(r'([0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])', value)
    need(match is not None, 'PCI_FULL_BDF_FORMAT')
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    need(domain <= 0xffff and bus <= 0xff and device <= 0x1f and function <= 7, 'PCI_BDF_RANGE')
    return '%04x:%02x:%02x.%x' % (domain, bus, device, function)


def tail_matches(current, old, rank):
    need(current['arm'] == old['arm'] == 'domain' and current['mode'] == 'resume20to25'
         and old['mode'] == 'fresh25' and current['pair_id'] == old['pair_id'] == PAIR, 'OWN_DOMAIN_PAIR')
    inputs, forwards, losses = (current[k] for k in ('training_inputs','forward_inputs','losses'))
    need(len(old['training_inputs']) == len(old['forward_inputs']) == len(old['losses']) == 100
         and len(inputs) == len(forwards) == len(losses) == 20, 'CONTINUOUS100_RESUMED20_DENOMINATOR')
    keys = ('global_step_before','canonical_ordinal','source_id_SHA256','collated_fields_SHA256','training_inputs_SHA256',
            'input_tokens','effective_supervised_tokens','supervised_EOS_count')
    for i, (row, forward, loss) in enumerate(zip(inputs, forwards, losses)):
        expected, expected_forward = old['training_inputs'][80+i], old['forward_inputs'][80+i]
        need(row['microbatch'] == forward['microbatch'] == i and row['global_step_before'] == 20+i//4
             and all(row[k] == expected[k] for k in keys), 'ACTUAL_CONTINUOUS_TAIL_INPUT')
        need(forward['input_SHA256'] == expected_forward['input_SHA256']
             and forward['RNG_before'] == expected_forward['RNG_before'] and forward['RNG_before']['finite'] is True,
             'ACTUAL_CONTINUOUS_FORWARD_RNG')
        need(type(loss) in (int,float) and math.isfinite(loss) and loss == old['losses'][80+i], 'ACTUAL_EXACT_CONTINUOUS_LOSS')
    need(current['actual_input_tokens'] == sum(row['input_tokens'] for row in inputs)
         and current['actual_supervised_tokens'] == sum(row['effective_supervised_tokens'] for row in inputs), 'ACTUAL_TAIL_TOKEN_SUM')
    return dict(rank=rank, input_tokens=current['actual_input_tokens'], supervised_tokens=current['actual_supervised_tokens'],
                microbatches=20, real_new_updates=5, input_forward_RNG_loss_tail_exact=True)


def audit(bundle, context, old_ledger):
    need(bundle['context'] == context and bundle['remote_writes'] == bundle['Torch_GPU_calls'] == bundle['state_body_reads'] == 0,
         'READONLY_ACTUAL_CONTEXT')
    files, missing, errors = bundle['files'], [], []
    result = dict(request_id=context['request_id'], publication=context['publication'], epoch=context['epoch'],
        original_execution_state=bundle['original_execution_state'], planned_ranks=8, planned_microbatches_per_rank=20,
        planned_global_microbatches=160, planned_new_updates_per_rank=5, planned_global_steps=[21,22,23,24,25],
        observed_microbatches=0, actual_complete_rank_ledgers=0, completed_rank_receipts=0, missing_rank_receipts=[],
        engineering_metadata_verified=False, clean_post=False, exact_recovery_producer_metadata_verified=False,
        independently_reloaded_typed_state=False, actual_new_GPU_calls=0, model_or_state_body_reads=0,
        original_domain_FAIL_preserved=True, original_domain_missing_rank0_receipt_preserved=True,
        historical_domain_final_optimizer_live_digest_missing=True, original_domain_replay_allowed=False,
        Hygon_qualified=False, formal17=False, performance_qualified=False, effect_qualified=False, replay_allowed=False)

    def get(name):
        row = files.get(name)
        if not row or row.get('exists') is False:
            missing.append(name); return None
        return row['value']

    def check(name, operation):
        try:
            return operation()
        except (ValueError, KeyError, TypeError, AssertionError, IndexError) as error:
            errors.append(dict(boundary=name,error_type=type(error).__name__,reason=str(error)[:160])); return None

    outcome = get('run/outcome.json')
    if outcome is None or bundle['original_execution_state'] == 'UNKNOWN':
        result.update(missing_files=missing, errors=errors); return result
    config = get('run/domain-config.json')

    def original():
        need(outcome['request_id'] == context['request_id'] and outcome['source_publication'] == context['publication']
             and outcome['epoch'] == context['epoch'] and outcome['deadline'] == context['epoch']+900
             and outcome['finished_epoch'] <= outcome['deadline'] and outcome['arm'] == 'domain'
             and outcome['mode'] == 'resume20to25' and outcome['pair_id'] == PAIR, 'ACTUAL_RESUME_REQUEST')
        need(config['request_id'] == context['request_id'] and config['source_publication'] == context['publication']
             and config['epoch'] == context['epoch'] and config['deadline'] == context['epoch']+820
             and config['mode'] == 'resume20to25' and config['arm'] == 'domain'
             and config['physical_rows'] == 800 and config['world_size'] == 8, 'ACTUAL_RESUME_CONFIG')
        process = bundle['original_process_observation']
        need(process['observer_host'] == process['original_host'] and len(process['processes']) == 2
             and all(row['original_process_live'] is False for row in process['processes']), 'ORIGINAL_OWNED_PROCESS_STILL_UNKNOWN')
        source = get('src/SOURCE_FILES.json')
        need(files['src/SOURCE_FILES.json']['SHA256'] == context['source_manifest_SHA256'], 'ACTUAL_SOURCE_MF')
        for name in ('CLAIM.json','CONFIG.template.json','INSTALL_PLAN.json','MERGED_FRAMEWORK_FILES.json','supervisor/SOURCE_FILES.json','supervisor/CLAIM.json'):
            need(files['src/'+name]['bytes'] == source['files'][name]['bytes']
                 and files['src/'+name]['SHA256'] == source['files'][name]['SHA256'], 'ACTUAL_SOURCE_MEMBER')
        nested = get('src/supervisor/SOURCE_FILES.json')
        need(config['owned_cleanup']['supervisor_source']['SHA256'] == nested['files']['bounded_GPU_supervisor.py']['SHA256']
             == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'ACTUAL_RSS_V2_PARENT')
    check('original-source', original)

    def post_gate():
        post = get('run/after.json'); cleanup = outcome['cleanup']
        need(post == outcome['post'] and post['active_compute_processes'] == 0 and post['master_port_bindable'] is True
             and len(post['devices']) == 8 and {(r['UUID'],canonical_PCI(r['PCI'])) for r in post['devices']}
             == {(r['UUID'],canonical_PCI(r['PCI'])) for r in config['expected_GPU_inventory']}, 'SAME8_POST')
        need(cleanup['owned_processes_remaining'] == 0 and cleanup['held_child']['reaped'] is True
             and outcome['release_permitted'] is True, 'OWNED_POST_CLEANUP')
        return True
    result['clean_post'] = bool(check('post', post_gate))

    def resource_gate():
        resource = get('run/resource-samples.json')
        need(resource == outcome['resources'] and resource['soft_limit_exceeded'] is False
             and resource['hard_limit'] is False and resource['soft_limit_bytes'] == 256*1024**3, 'RSS_V2_RECORD')
        return dict(sampled_group_peak_RSS_bytes=resource['sampled_group_peak_RSS_bytes'],samples=resource['samples'],hard_RSS=False)
    result['resources'] = check('resources', resource_gate)
    ranks, adapters, schedulers = [], set(), set()
    for rank in range(8):
        prefix = 'run/witness/rank-%02d/'%rank
        owner, progress, receipt, micro = (get(prefix+n) for n in ('owner.json','progress.json','receipt.json','microbatches.json'))
        count = len(micro.get('training_inputs',[])) if isinstance(micro,dict) else 0
        completed = progress.get('completed_microbatches') if isinstance(progress,dict) else None
        valid_completed = type(completed) is int and 0 <= completed <= 20
        result['observed_microbatches'] += completed if valid_completed else count
        if receipt is None:
            result['missing_rank_receipts'].append(rank)
        else:
            result['completed_rank_receipts'] += 1
        row = dict(rank=rank,observed_microbatches=completed if valid_completed else count,
                   saved_ledger_microbatches=count,receipt_present=receipt is not None,verified=False)
        loss_observation = get(prefix+'tail-loss-observation.json')
        if isinstance(loss_observation,dict):
            row['last_actual_loss_observation'] = {key:loss_observation.get(key) for key in
                ('microbatch','global_step_before','expected_loss','actual_loss','exact','planned_microbatches',
                 'attempted_microbatches','completed_native_returns','native_training_step_called','native_training_step_returned_finite')}
        ranks.append(row)

        def rank_gate():
            need(owner['request_id'] == context['request_id'] and owner['rank'] == rank and owner['pair_id'] == PAIR
                 and owner['mode'] == 'resume20to25' and owner['arm'] == 'domain' and owner['epoch'] == context['epoch']
                 and owner['deadline'] == context['epoch']+820 and owner['PID'] > 0 and owner['startticks'] > 0, 'RESUME_OWNER')
            need(progress['completed_microbatches'] == 20 and progress['completed_updates'] == [21,22,23,24,25]
                 and progress['saved_steps'] == [25] and progress['finished'] is True and progress['error'] is None, 'FULL_RESUME_PROGRESS')
            need(receipt['rank'] == rank and receipt['mode'] == 'resume20to25' and receipt['arm'] == 'domain'
                 and receipt['pair_id'] == PAIR and receipt['completed_microbatches'] == 20 and receipt['updates'] == 5
                 and receipt['completed_step'] == 25 and receipt['recovery_exact'] is True
                 and files[prefix+'receipt.json']['SHA256'] == outcome['rank_receipts'][str(rank)]['file']['SHA256'], 'RESUME_RECEIPT')
            restored = get(prefix+'restored-state-before-rng.json')
            need(all(restored[k]['exact'] is True for k in ('adapter','optimizer','scheduler')), 'RESTORED20_TYPED_PRODUCER')
            rng = get(prefix+'rng-restoration-point.json')
            need(rng['phase'] == 'NATIVE_LOAD_RNG_CALL_RETURN' and rng['exact'] is True and rng['global_step'] == 20
                 and rng['current_SHA256'] == rng['expected_SHA256'], 'NATIVE_RNG_RESTORATION_POINT')
            saved = old_ledger['files']['witness/rank-%02d/microbatches.json'%rank]
            pin = config['continuous_reference'][str(rank)]
            need(saved['bytes'] == pin['bytes'] and saved['SHA256'] == pin['SHA256'], 'ORIGINAL_LOCAL_CONTINUOUS100_PIN')
            compared = tail_matches(micro,saved['value'],rank)
            loss = loss_observation
            need(loss['microbatch'] == 19 and loss['global_step_before'] == 24 and loss['planned_microbatches'] == 20
                 and loss['completed_native_returns'] == 20 and loss['exact'] is True
                 and loss['actual_loss'] == loss['expected_loss'] == micro['losses'][-1], 'FINAL_LOSS_OBSERVATION')
            end = get(prefix+'resume25-comparison.json')
            need(all(end[k]['exact'] is True for k in ('adapter','optimizer','scheduler'))
                 and end['saved_rank_RNG_exact'] is True and end['loss_tail_exact'] is True and end['loss_tail_count'] == 20, 'FINAL25_ALL_STATE_EXACT')
            after, scheduler = get(prefix+'update-25-after.json'), get(prefix+'update-25-scheduler.json')
            need(after['optimizer_steps'] == scheduler['global_step'] == 25 and scheduler['epoch'] == 1., 'FINAL25_STEP')
            adapters.add(after['adapter_SHA256']); schedulers.add(scheduler['scheduler_SHA256'])
            row.update(compared,verified=True,restored20_typed_producer_exact=True,RNG_restore_point_exact=True,
                       final25_typed_and_saved_rank_RNG_producer_exact=True)
            return True
        if check('rank-%02d'%rank,rank_gate):
            result['actual_complete_rank_ledgers'] += 1
    result['ranks'] = ranks
    def all8():
        need(result['actual_complete_rank_ledgers'] == 8 and result['observed_microbatches'] == 160
             and result['completed_rank_receipts'] == 8 and len(adapters) == len(schedulers) == 1, 'ALL8_160_5_UPDATES')
        return dict(actual_global_microbatches=160,actual_new_updates_per_rank=5,final_global_step=25,
                    actual_input_tokens=sum(row['input_tokens'] for row in ranks),
                    actual_supervised_tokens=sum(row['supervised_tokens'] for row in ranks),
                    live_adapter_all8_SHA256=next(iter(adapters)),live_scheduler_all8_SHA256=next(iter(schedulers)))
    summary = check('all8',all8)
    if summary:
        result.update(summary)
    name = 'run/witness/rank-00/checkpoint-25-files.json'; checkpoint = get(name)
    def saved25():
        need(checkpoint['path'].endswith('/run/train/checkpoint-25') and set(checkpoint['files']) == MEMBERS
             and all(pin['bytes'] > 0 and len(pin['SHA256']) == 64 for pin in checkpoint['files'].values()), 'SAVED_FULL14_CHECKPOINT25')
        return dict(files=14,bytes=sum(pin['bytes'] for pin in checkpoint['files'].values()),manifest_SHA256=files[name]['SHA256'])
    result['checkpoint25'] = check('checkpoint25',saved25)
    result.update(errors=errors,missing_files=sorted(set(missing)),
        engineering_metadata_verified=bool(not errors and not missing and outcome['engineering_completed'] is True
            and outcome['native_returncode'] == 0 and not outcome['cleanup']['termination_needed'] and result['clean_post']),
        exact_recovery_producer_metadata_verified=bool(summary and not errors and not missing),
        typed_evidence_boundary='Frozen producer safe-load comparisons plus saved JSON/receipt, not new independent tensor/body reload.',
        source_condition_saved25_equals_live='Rank0 frozen finish requires saved25 typed state equal live before final receipt.',
        original_small_files={name:{k:row.get(k) for k in ('bytes','SHA256','exists')} for name,row in files.items()})
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('bundle','bundle-sha256','context','old-ledger','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    path = Path(args.bundle); need(path.stat().st_size <= 8*1024**2,'NEW_SMALL_BUNDLE_CAP')
    raw = path.read_bytes(); need(H(raw) == args.bundle_sha256,'ACTUAL_FIRST_NEW_BUNDLE_PIN')
    prior = Path(args.old_ledger).read_bytes()
    need(len(prior) == 603701 and H(prior) == '7e34e226ef08d879f4f1ff6c982ead01495e850e95be97fd3ab94b679b8877dc','OLD_LOCAL_LEDGER_EXACT_PIN')
    result = audit(json.loads(raw),json.loads(Path(args.context).read_bytes()),json.loads(prior))
    with Path(args.output).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('original_execution_state','observed_microbatches','completed_rank_receipts',
          'engineering_metadata_verified','clean_post','formal17')}))


if __name__ == '__main__':
    main()
