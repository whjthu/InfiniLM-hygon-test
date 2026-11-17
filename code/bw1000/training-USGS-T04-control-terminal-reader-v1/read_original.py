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


def audit(bundle, context):
    need(bundle['context'] == context and bundle['remote_writes'] == bundle['Torch_GPU_calls'] == bundle['state_body_reads'] == 0,
         'READONLY_ORIGINAL_CONTEXT')
    files, errors, missing = bundle['files'], [], []

    def get(name):
        row = files.get(name)
        if not row or row.get('exists') is False:
            missing.append(name)
            return None
        need(type(row['bytes']) is int and 0 < row['bytes'] <= 2*1024**2 and len(row['SHA256']) == 64,
             'ORIGINAL_SMALL_PIN_SCHEMA')
        return row['value']

    def check(name, operation):
        try:
            return operation()
        except (ValueError, KeyError, TypeError, AssertionError, IndexError, ZeroDivisionError) as error:
            errors.append(dict(boundary=name, error_type=type(error).__name__, reason=str(error)[:160]))
            return None

    outcome = get('run/outcome.json')
    result = dict(request_id=context['request_id'], publication=context['publication'], epoch=context['epoch'],
        original_execution_state=bundle['original_execution_state'], planned_ranks=8, planned_global_microbatches=800,
        planned_updates_per_rank=25, planned_microbatches_per_rank=100, actual_complete_rank_ledgers=0,
        observed_microbatches=0, completed_rank_receipts=0, missing_rank_receipts=[], clean_post=False,
        independent_saved_JSON_consumption=False, engineering_metadata_verified=False, full_training_qualified=False,
        formal17=False, Hygon_qualified=False, performance_qualified=False, effect_qualified=False,
        checkpoint_restore_qualified=False, actual_new_GPU_calls=0, model_state_body_reads=0,
        initial_trainables_independently_rehashed=False, original_domain_FAILURE_preserved=True,
        original_domain_replay_allowed=False, replay_allowed=False)
    if not outcome or bundle['original_execution_state'] == 'UNKNOWN':
        result.update(errors=errors, missing_files=missing)
        return result
    config = get('run/domain-config.json')
    def original():
        need(outcome['request_id'] == context['request_id'] and outcome['source_publication'] == context['publication']
             and outcome['epoch'] == context['epoch'] and outcome['deadline'] == context['epoch']+900
             and outcome['finished_epoch'] <= outcome['deadline'] and outcome['arm'] == 'generic'
             and outcome['mode'] == 'fresh25' and outcome['pair_id'] == PAIR, 'ACTUAL_CONTROL_REQUEST_EPOCH')
        need(config['request_id'] == context['request_id'] and config['source_publication'] == context['publication']
             and config['epoch'] == context['epoch'] and config['deadline'] == context['epoch']+820
             and config['arm'] == 'generic' and config['mode'] == 'fresh25'
             and config['pair_id'] == PAIR and config['world_size'] == 8 and config['physical_rows'] == 800,
             'ACTUAL_CONTROL_CONFIG')
        process = bundle['original_process_observation']
        need(process['observer_host'] == process['original_host'] and len(process['processes']) == 2
             and all(row['original_process_live'] is False for row in process['processes']), 'ORIGINAL_PROCESSES_NOT_CONFIRMED_ABSENT')
    check('original', original)

    def source_gate():
        mf = get('src/SOURCE_FILES.json')
        need(files['src/SOURCE_FILES.json']['SHA256'] == context['source_manifest_SHA256'], 'ORIGINAL_ACTUAL_SOURCE_MF')
        for name in ('CLAIM.json','CONFIG.template.json','INSTALL_PLAN.json','MERGED_FRAMEWORK_FILES.json',
                     'supervisor/SOURCE_FILES.json','supervisor/CLAIM.json'):
            row = files['src/'+name]; need(row['bytes'] == mf['files'][name]['bytes']
                 and row['SHA256'] == mf['files'][name]['SHA256'], 'ACTUAL_SOURCE_MEMBER_PIN')
        merged = get('src/MERGED_FRAMEWORK_FILES.json')
        need(len(merged['files']) == 400 and config['framework_manifest']['SHA256'] == files['src/MERGED_FRAMEWORK_FILES.json']['SHA256'], 'COMMON400')
        nested = get('src/supervisor/SOURCE_FILES.json')
        need(config['owned_cleanup']['supervisor_source']['SHA256'] == nested['files']['bounded_GPU_supervisor.py']['SHA256']
             == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'ACTUAL_PARENT_RSS_V2_SOURCE')
    check('source', source_gate)

    def post_gate():
        post = get('run/after.json'); cleanup = outcome['cleanup']
        need(post == outcome['post'] and post['active_compute_processes'] == 0 and post['master_port_bindable'] is True
             and len(post['devices']) == 8 and len({row['UUID'] for row in post['devices']}) == 8
             and len({row['PCI'] for row in post['devices']}) == 8, 'ACTUAL_POST')
        expected = config['expected_GPU_inventory']
        need({(row['UUID'], canonical_PCI(row['PCI'])) for row in post['devices']}
             == {(row['UUID'], canonical_PCI(row['PCI'])) for row in expected}, 'POST_ORIGINAL8_DEVICES')
        need(cleanup['owned_processes_remaining'] == 0 and cleanup['held_child']['reaped'] is True
             and outcome['release_permitted'] is True, 'OWNED_CLEANUP')
        return dict(clean=True, termination_needed=cleanup['termination_needed'])
    post = check('post', post_gate)
    if post:
        result.update(clean_post=post['clean'], termination_needed=post['termination_needed'])

    def resource_gate():
        resource = get('run/resource-samples.json')
        need(resource == outcome['resources'] and resource['hard_limit'] is False
             and resource['soft_limit_bytes'] == 256*1024**3 and resource['soft_limit_exceeded'] is False, 'RSS_V2_RECORD')
        nested = get('src/supervisor/SOURCE_FILES.json')
        need(nested['files']['bounded_GPU_supervisor.py']['SHA256'] == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f', 'ACTUAL_RSS_V2_SOURCE')
        return dict(sampled_group_peak_RSS_bytes=resource['sampled_group_peak_RSS_bytes'], samples=resource['samples'],
                    RSS_hard_limit=False, coverage='owned sampled VmRSS; UNKNOWN rows retained by actual RSS-v2 record')
    resources = check('resources', resource_gate)
    if resources:
        result['resources'] = resources

    def references():
        for name in ('native45', 'training800', 'occurrences', 'pair_initial_reference'):
            row = files['reference/'+name]; pin = config[name]
            need(row['bytes'] == pin['bytes'] and row['SHA256'] == pin['SHA256'], 'EXACT_REFERENCE_PIN_'+name)
        need(config['training800']['bytes'] == 314277 and config['training800']['SHA256'] == '564e7439a41de9d6c8b76af2ae0392f154e5157c60687aaff12e970f593986ab', 'CONTROL800')
        need(config['initial_state']['bytes'] == 174808783 and config['initial_state']['SHA256'] == INITIAL_SHA
             and config['initial_state']['named_SHA256'] == NAMED_SHA, 'ORIGINAL_DOMAIN_INITIAL')
        ref = get('reference/pair_initial_reference')
        need(ref['initial_state'] == config['initial_state'] and ref['pair_id'] == PAIR and ref['rank'] == ref['global_step'] == 0
             and ref['arm'] == 'domain' and ref['tensors'] == 504 and ref['elements'] == 43646976, 'ORIGINAL_PAIR_REFERENCE')
        native = [r for r in get('reference/native45') if r['split'] == 'train']
        need(len(native) == 45 and [row['ordinal'] for row in native] == list(range(45)), 'NATIVE45')
        view = get('reference/training800'); occurrence = get('reference/occurrences')
        indices = occurrence['original_index_by_physical_row']
        need(len(view) == len(indices) == 800 and occurrence['arm'] == 'generic'
             and occurrence['view_SHA256'] == config['training800']['SHA256'], 'CONTROL800_OCCURRENCES')
        need(all(type(i) is int and 0 <= i < 45 and H(row['id'].encode()) == native[i]['source_id_SHA256'] for row, i in zip(view, indices)), 'CONTROL800_SOURCE_IDS')
        return native, collections.Counter(indices)
    refs = check('references', references)
    counters, rank_rows, adapter, scheduler = collections.Counter(), [], set(), set()
    for rank in range(8):
        prefix = 'run/witness/rank-%02d/'%rank
        owner, progress, receipt, first = (get(prefix+name) for name in ('owner.json', 'progress.json', 'receipt.json', 'initial-state.json'))
        micro = get(prefix+'microbatches.json')
        if receipt is None:
            result['missing_rank_receipts'].append(rank)
        else:
            result['completed_rank_receipts'] += 1
        observed = len(micro.get('training_inputs', [])) if isinstance(micro, dict) else 0
        result['observed_microbatches'] += observed
        row = dict(rank=rank, observed_microbatches=observed, receipt_present=receipt is not None, verified=False)
        rank_rows.append(row)

        def rank_gate():
            need(owner['rank'] == rank and owner['request_id'] == context['request_id'] and owner['pair_id'] == PAIR
                 and owner['arm'] == 'generic' and owner['epoch'] == context['epoch']
                 and owner['deadline'] == context['epoch']+820 and owner['PID'] > 0 and owner['startticks'] > 0, 'ACTUAL_OWNER')
            need(progress['completed_microbatches'] == 100 and progress['completed_updates'] == list(range(1,26))
                 and progress['saved_steps'] == [10,20,25] and progress['finished'] is True and progress['error'] is None, 'FULL_PROGRESS')
            need(receipt['rank'] == rank and receipt['arm'] == 'generic' and receipt['pair_id'] == PAIR
                 and receipt['completed_microbatches'] == 100 and receipt['updates'] == receipt['completed_step'] == 25
                 and files[prefix+'receipt.json']['SHA256'] == outcome['rank_receipts'][str(rank)]['file']['SHA256'], 'ACTUAL_RECEIPT')
            need(first['arm'] == 'generic' and first['rank'] == rank and first['global_step'] == 0 and first['RNG_restored'] is False
                 and first['tensors'] == 504 and first['elements'] == 43646976 and first['named_SHA256'] == NAMED_SHA
                 and isinstance(first['restoration'], dict) and bool(first['restoration']), 'INITIAL_VALUES_ACTUALLY_RESTORED')
            inputs, forwards, losses = (micro[key] for key in ('training_inputs', 'forward_inputs', 'losses'))
            need(micro['arm'] == 'generic' and micro['mode'] == 'fresh25' and micro['pair_id'] == PAIR
                 and len(inputs) == len(forwards) == len(losses) == 100
                 and all(type(loss) in (int,float) and math.isfinite(loss) for loss in losses), 'FULL_FINITE100')
            need(refs is not None, 'NATIVE_REFERENCE_UNQUALIFIED')
            native, _ = refs; local = collections.Counter()
            for i, (item, forward) in enumerate(zip(inputs, forwards)):
                ordinal = item['canonical_ordinal']; need(type(ordinal) is int and 0 <= ordinal < 45, 'ORDINAL')
                reference = native[ordinal]
                need(item['microbatch'] == forward['microbatch'] == i and item['global_step_before'] == i//4
                     and item['source_id_SHA256'] == reference['source_id_SHA256']
                     and item['collated_fields_SHA256'] == fingerprint(reference['collated_fields'])
                     and item['input_tokens'] == reference['full_length'] and item['effective_supervised_tokens'] > 0
                     and item['supervised_EOS_count'] == reference['supervised_EOS_count'] and forward['RNG_before']['finite'] is True,
                     'ACTUAL_NATIVE_INPUT_LABEL_EOS')
                if 'effective_supervised_tokens' in reference:
                    need(item['effective_supervised_tokens'] == reference['effective_supervised_tokens'], 'NATIVE_SUPERVISED_COUNTER')
                local[ordinal] += 1
            need({str(k):v for k,v in local.items()} == micro['canonical_occurrences'], 'LOCAL_OCCURRENCES')
            it = sum(x['input_tokens'] for x in inputs); st = sum(x['effective_supervised_tokens'] for x in inputs)
            need(it == micro['actual_input_tokens'] and st == micro['actual_supervised_tokens'], 'ACTUAL_TOKENS')
            after = get(prefix+'update-25-after.json'); sched = get(prefix+'update-25-scheduler.json')
            need(after['optimizer_steps'] == sched['global_step'] == 25 and sched['epoch'] == 1., 'FINAL25')
            for step in (1,2):
                delta = get(prefix+'first-update-%d-actual-delta.json'%step)
                need(delta['global_step'] == delta['actual_optimizer_step'] == step and delta['optimizer_state_entries'] == 504
                     and delta['parameter_delta']['parameter_tensors'] == 504, 'ACTUAL_FIRST2_UPDATE')
                if step == 1:
                    need(all(lr == 0 for lr in delta['learning_rates']) and delta['parameter_delta']['changed_elements'] == 0, 'ORIGINAL_FIRST_LR0')
                else:
                    need(delta['parameter_delta']['changed_elements'] > 0, 'REAL_NONZERO_PARAMETER_UPDATE')
            adapter.add(after['adapter_SHA256']); scheduler.add(sched['scheduler_SHA256']); counters.update(local)
            row.update(verified=True, input_tokens=it, supervised_tokens=st, actual_initial_restore_record_present=True)
            return True
        if check('rank-%02d'%rank, rank_gate):
            result['actual_complete_rank_ledgers'] += 1
    result['ranks'] = rank_rows

    def global_gate():
        need(refs is not None and result['actual_complete_rank_ledgers'] == 8 and sum(counters.values()) == 800
             and counters == refs[1] and len(adapter) == len(scheduler) == 1, 'ALL8_CONTROL800')
        for rank in range(8):
            counts = get('run/witness/rank-%02d/global800-occurrences.json'%rank)
            need(counts == dict(counts=[counters[i] for i in range(45)], total=800), 'ALL8_REDUCED_COUNTS')
        return dict(actual_global_microbatches=800, actual_unique_rows=45, minimum_exposures=min(counters.values()),
                    maximum_exposures=max(counters.values()), live_adapter_SHA256=next(iter(adapter)),
                    live_scheduler_SHA256=next(iter(scheduler)), input_tokens=sum(r['input_tokens'] for r in rank_rows),
                    supervised_tokens=sum(r['supervised_tokens'] for r in rank_rows))
    complete = check('global800', global_gate)
    if complete:
        result.update(complete)

    checkpoints = {}
    for step in (10,20,25):
        name = 'run/witness/rank-00/checkpoint-%d-files.json'%step
        value = get(name)
        def checkpoint_gate():
            need(value['path'].endswith('/run/train/checkpoint-'+str(step)) and set(value['files']) == MEMBERS
                 and all(type(pin['bytes']) is int and pin['bytes'] > 0 and len(pin['SHA256']) == 64 for pin in value['files'].values()), 'FULL14_CHECKPOINT_MANIFEST')
            return dict(files=14, bytes=sum(x['bytes'] for x in value['files'].values()), manifest_SHA256=files[name]['SHA256'])
        checkpoints[str(step)] = check('checkpoint-'+str(step), checkpoint_gate)
    result['checkpoints'] = checkpoints
    result.update(errors=errors, missing_files=sorted(set(missing)),
        independent_saved_JSON_consumption=True,
        engineering_metadata_verified=bool(not errors and not missing and outcome['engineering_completed'] is True
            and outcome['native_returncode'] == 0 and not outcome['cleanup']['termination_needed'] and result['clean_post']),
        initial_RNG_before_after='Producer frozen plugin asserts unchanged; actual before/after raw RNG snapshots not independently stored.',
        saved25_equals_live='Producer frozen rank0 finish asserts typed saved25==live before final receipt. No new independent typed body read.',
        historical_domain_initial_RNG_crossarm_exact='UNKNOWN_NOT_INDEPENDENTLY_SAVED_AS_SEPARATE_INITIAL_SNAPSHOTS',
        original_small_files={name:dict(bytes=row.get('bytes'), SHA256=row.get('SHA256'), exists=row.get('exists',True)) for name,row in files.items()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.bundle); need(path.stat().st_size <= 8*1024**2, 'SAVED_BUNDLE_BOUND')
    raw = path.read_bytes(); need(H(raw) == args.bundle_sha256, 'FIRST_ORIGINAL_BUNDLE_PIN')
    context = json.loads(Path(args.context).read_bytes())
    result = audit(json.loads(raw), context)
    with Path(args.output).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key:result[key] for key in ('original_execution_state','observed_microbatches','completed_rank_receipts',
          'engineering_metadata_verified','clean_post','formal17','performance_qualified')}))


if __name__ == '__main__':
    main()
