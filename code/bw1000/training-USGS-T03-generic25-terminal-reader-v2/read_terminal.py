"""Independent generic25 metadata audit; no tensor, input, image or installed-provider body reads."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import time

SCOPE = Path('/home/whj/nvidia-baseline/alignment-assets/training-USGS-T03-fresh-control-A100-v1')
MEMBERS = {'adapter_config.json', 'adapter_model.safetensors', 'args.json', 'optimizer.pt',
           'scheduler.pt', 'trainer_state.json'} | {'rng_state_%d.pth' % i for i in range(8)}
INITIAL_NAMED = '086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137'


def need(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set('0123456789abcdef')


def read_terminal(context, read_origin, reader_request):
    need(context['root_GO'] is True and context['READ_GO'] is True, 'ROOT_READER_ADOPTION_REQUIRED')
    need(type(read_origin) in (int, float) and math.isfinite(read_origin) and
         0 < read_origin <= time.time() < read_origin + 45, 'ONE_READER_EPOCH')
    need(len(reader_request) == 32 and not set(reader_request) - set('0123456789abcdef'), 'READER_REQUEST_ID')
    original = context['original']
    need(digest(original['source_manifest_SHA256']) and digest(original['claim_SHA256']) and
         len(original['request_id']) == 32 and len(original['publication']) == 40 and
         type(original['epoch']) in (int, float) and math.isfinite(original['epoch']), 'ORIGINAL_EXECUTION_CONTEXT')
    stop = time.monotonic() + read_origin + 45 - time.time()
    def check():
        need(time.time() < read_origin + 45 and time.monotonic() < stop, 'READER_ORIGINAL_EPOCH_EXHAUSTED')
    def expired(*_):
        raise TimeoutError('READER_ORIGINAL_EPOCH_EXHAUSTED')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, read_origin + 45 - time.time())
    allowed = sorted(os.sched_getaffinity(0) - {0, 1}); need(len(allowed) >= 2, 'TWO_CPU_REQUIRED')
    os.sched_setaffinity(0, set(allowed[:2])); need(os.sched_getaffinity(0) == set(allowed[:2]), 'ACTUAL_READER_CPU_MASK')
    pins = {}; cache = {}; total = 0; observed_ranks = []; complete = 0
    result = dict(schema='T03_PAIRED_GENERIC25_INDEPENDENT_TERMINAL_V1', case='T03.sft', arm='generic',
        namespace=str(SCOPE), pair_id=None, request_id=original['request_id'], publication=original['publication'],
        epoch=original['epoch'], source_manifest_SHA256=original['source_manifest_SHA256'],
        claim_SHA256=original['claim_SHA256'], independent_passed=False, engineering_completed=False,
        release_permitted=False, planned_ranks=8, planned_rank_microbatches=100, planned_updates=25,
        planned_global_micro_exposures=800, completed_ranks=0, completed_updates=0, global_micro_exposures=0,
        resource_post_observed_clean=None, Model_GO=False, formal17_GO=False, effect_qualified=False,
        performance_qualified=False, new_training_calls=0, new_preprocessing_calls=0, new_GPU_queries=0,
        tensor_or_weight_bodies_read=False, train_dev_blind_gold_or_predictions_read=False)
    def metadata_path(name):
        if name.startswith('views/'):
            return Path('/home/whj/nvidia-baseline/alignment-assets/training-USGS-T03-fresh-domain-A100-v1') / name
        return SCOPE / name
    def load(name, optional=False):
        nonlocal total
        if name in cache:
            return cache[name]
        check(); p = metadata_path(name)
        try:
            before = p.lstat()
        except FileNotFoundError:
            if optional:
                pins[name] = dict(exists=False); cache[name] = None; return None
            raise
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 2 * 1024**2, 'BOUNDED_REGULAR_METADATA')
        with p.open('rb') as f:
            raw = f.read(2 * 1024**2 + 1); after = os.fstat(f.fileno())
        key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(key(before) == key(after) == key(p.lstat()) and len(raw) == before.st_size, 'STABLE_METADATA_IDENTITY')
        total += len(raw); need(total <= 16 * 1024**2, 'TOTAL_METADATA16MIB')
        pins[name] = dict(bytes=len(raw), SHA256=sha(raw))
        cache[name] = raw
        return raw
    def obj(name, optional=False):
        raw = load(name, optional)
        return json.loads(raw) if raw is not None else None
    def metadata_pin(name):
        return dict(path=str(metadata_path(name)), **pins[name])
    def same_context(row, *, source_key='publication'):
        need(row['request_id'] == original['request_id'] and row[source_key] == original['publication'] and
             row['epoch'] == original['epoch'], 'ORIGINAL_REQUEST_PUBLICATION_EPOCH')
    def absent(saved):
        need(type(saved['PID']) is int and saved['PID'] > 0 and
             type(saved['startticks']) is int and saved['startticks'] > 0, 'STRONG_ORIGINAL_IDENTITY')
        try:
            fields = (Path('/proc') / str(saved['PID']) / 'stat').read_text().rsplit(') ', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            return 'ORIGINAL_PID_ABSENT'
        need(int(fields[19]) != saved['startticks'], 'ORIGINAL_PROCESS_STILL_LIVE')
        return 'ORIGINAL_IDENTITY_ABSENT_PID_REUSED'
    def pci(value):
        parts = value.lower().split(':'); need(len(parts) == 3, 'PCI_SCHEMA')
        return '%04x:%02x:%s' % (int(parts[0], 16), int(parts[1], 16), parts[2])
    def finish():
        result.update(reader_request=reader_request, reader_origin=read_origin, reader_finished=time.time(),
            reader_CPU_affinity=sorted(os.sched_getaffinity(0)), metadata_file_count=len(pins),
            metadata_existing_files_read=sum('bytes' in pin for pin in pins.values()), metadata_bytes_read=total,
            metadata_trace_SHA256=sha(canonical(pins)), metadata_trace_includes_missing=True,
            critical_pins={name: pins[name] for name in ('src/SOURCE_FILES.json', 'src/CLAIM.json', 'install-intent.json',
                'install-ready.json', 'views/pair-views-private.json', 'run/domain-config.json', 'run/outcome.json',
                'run/after.json', 'run/common-entry-receipt.jsonl') if name in pins},
            proof_boundary='Independent metadata/source-identity/denominator closure. Actual state values were safely reloaded and compared by the frozen native producer; this reader never reloads tensors. Resource cleanup is separate from model/effect/formal acceptance.')
        need(len(canonical(result)) <= 16384, 'BOUNDED_RESULT16KIB')
        return result
    try:
        mf = obj('src/SOURCE_FILES.json')
        need(pins['src/SOURCE_FILES.json']['SHA256'] == original['source_manifest_SHA256'], 'TOP_LEVEL_PRODUCER_SOURCE_MF')
        claim = obj('src/CLAIM.json'); plan = obj('src/INSTALL_PLAN.json'); initials = obj('src/INITIAL_REFERENCES.json')
        need(pins['src/CLAIM.json']['SHA256'] == original['claim_SHA256'] and claim['case'] == 'T03.sft' and
             claim['root_GO'] is True and claim['GPU_GO'] is True and claim['max_invocations_per_arm'] == 1,
             'ORIGINAL_ADOPTED_T03_SCOPE')
        for name in ('CLAIM.json', 'INSTALL_PLAN.json', 'INITIAL_REFERENCES.json'):
            need(pins['src/' + name] == mf['files'][name], 'SOURCE_METADATA_MEMBER_PIN')
        nested = obj('src/supervisor/SOURCE_FILES.json')
        need(pins['src/supervisor/SOURCE_FILES.json'] == mf['files']['supervisor/SOURCE_FILES.json'] == plan['supervisor_manifest'] and
             pins['src/supervisor/SOURCE_FILES.json']['SHA256'] == original['nested_source_manifest_SHA256'] and
             nested['files']['CLAIM.json'] == plan['supervisor_claim'] and
             plan['supervisor_claim']['SHA256'] == original['nested_claim_SHA256'] and
             nested['files']['qualified_driver.py']['SHA256'] == '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f',
             'NESTED_DRIVER_MF_DISTINCT_FROM_TOP_MF')
        result['pair_id'] = claim['pair_id']
        need(plan['namespaces']['generic'] == str(SCOPE) and claim['pair_id'] == plan['pair_id'], 'EXACT_DOMAIN_NAMESPACE')
        install = obj('install-intent.json'); same_context(install)
        need(install['source_manifest_SHA256'] == original['source_manifest_SHA256'] and
             install['claim_SHA256'] == original['claim_SHA256'] and install['pair_id'] == claim['pair_id'] and
             install['arm'] == 'generic' and install['deadline'] == original['epoch'] + 900 and
             install['UID'] == os.getuid() and install['host'] == os.uname().nodename and
             install['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'INSTALL_STRONG_CONTEXT')
        result['original_supervisor_terminal'] = absent(install)
        install_failure = obj('install-failure.json', optional=True)
        if install_failure is not None:
            need(install_failure['request_id'] == original['request_id'] and install_failure['epoch'] == original['epoch'] and
                 install_failure['state'] == 'KNOWN_INSTALL_FAILURE_NO_REPLAY', 'KNOWN_INSTALL_FAILURE_CONTEXT')
            result.update(state='KNOWN_INSTALL_FAILURE_RETAINED_NOT_TRAINING_PASS', independent_passed=True,
                          producer_failure_retained=True, failure_type=install_failure['error_type'])
            return finish()
        ready = obj('install-ready.json')
        need(ready['request_id'] == original['request_id'] and ready['epoch'] == original['epoch'] and
             ready['arm'] == 'generic' and ready['framework_members'] == 404 and ready['deadline'] == original['epoch'] + 900,
             'ACTUAL404_INSTALL_READY')
        prerequisite = obj('domain-qualification-private.json')
        need(pins['domain-qualification-private.json'] == dict(bytes=8309, SHA256='cf3e82548f90343e620a9e9d2aa26e73702a8b3a123517f25d072e54d35c9bce') and
             prerequisite['schema'] == 'T03_PAIRED_DOMAIN25_INDEPENDENT_TERMINAL_V1' and prerequisite['arm'] == 'domain' and
             prerequisite['independent_passed'] is True and prerequisite['engineering_completed'] is True and
             prerequisite['release_permitted'] is True and prerequisite['completed_ranks'] == 8 and
             prerequisite['completed_updates'] == 25 and prerequisite['global_micro_exposures'] == 800 and
             prerequisite['pair_id'] == claim['pair_id'] and prerequisite['source_manifest_SHA256'] == original['source_manifest_SHA256'],
             'ACTUAL_DOMAIN_INDEPENDENT_PREREQUISITE_PIN')
        views = obj('views/pair-views-private.json')
        need(metadata_pin('views/pair-views-private.json') == ready['pair_views'], 'INSTALLED_PAIR_VIEWS_PIN')
        result['views_manifest'] = ready['pair_views']
        domain = views['generic']; generic = views['domain']
        need(domain == ready['view_binding'] and domain['arm'] == 'generic' and generic['arm'] == 'domain' and
             domain['physical_rows'] == generic['physical_rows'] == 800 and domain['unique_rows'] == generic['unique_rows'] == 45 and
             domain['original_index_sequence_SHA256'] == generic['original_index_sequence_SHA256'] and
             domain['new_encodes'] == generic['new_encodes'] == 0, 'FROZEN_PAIRED80045')
        occurrence = obj('views/generic/exposure-ledger-private.json')
        need(metadata_pin('views/generic/exposure-ledger-private.json') == domain['occurrences'] and
             occurrence['arm'] == 'generic' and occurrence['view_SHA256'] == domain['training800']['SHA256'], 'ACTUAL_OCCURRENCE_LEDGER')
        indices = occurrence['original_index_by_physical_row']
        need(len(indices) == 800 and all(type(v) is int and 0 <= v < 45 for v in indices) and
             Counter(indices) == Counter({i: domain['counts'][i] for i in range(45)}), 'EXACT800_SCHEDULE')
        out = obj('run/outcome.json'); same_context(out, source_key='source_publication')
        need(out['arm'] == 'generic' and out['mode'] == 'fresh25' and out['pair_id'] == claim['pair_id'] and
             out['deadline'] == original['epoch'] + 900 and out['planned_ranks'] == 8 and
             out['planned_microbatches_per_rank'] == 100 and out['planned_updates_per_rank'] == 25, 'ORIGINAL_OUTCOME_DENOMINATORS')
        need(out['control']['source_manifest_SHA256'] == original['nested_source_manifest_SHA256'] and
             out['control']['claim_SHA256'] == original['nested_claim_SHA256'], 'ACTUAL_NESTED_CONTROLLER_SOURCE')
        result['producer_engineering_completed'] = out['engineering_completed']
        result['original_scope_resource_release_reported'] = out['release_permitted']
        result['native_returncode'] = out['native_returncode']
        result['producer_failure'] = out['failure']
        for rank in range(8):
            progress = obj('run/witness/rank-%02d/progress.json' % rank, optional=True)
            if progress is not None:
                observed_ranks.append(dict(rank=rank, completed_microbatches=progress['completed_microbatches'],
                    completed_updates=len(progress['completed_updates']), finished=progress['finished']))
        if out['engineering_completed'] is not True:
            need(out['native_returncode'] is not None or out['control']['exec_request_attempted'] is False,
                 'ORIGINAL_TERMINAL_OUTCOME_UNKNOWN')
            result.update(state='KNOWN_PRODUCER_FAILURE_RETAINED_NOT_TRAINING_PASS', independent_passed=True,
                          producer_failure_retained=True, observed_rank_progress=observed_ranks)
            return finish()
        c = obj('run/domain-config.json')
        need(c['request_id'] == original['request_id'] and c['source_publication'] == original['publication'] and
             c['epoch'] == original['epoch'] and c['deadline'] == original['epoch'] + 820 and c['case'] == 'T03.sft' and
             c['arm'] == 'generic' and c['mode'] == 'fresh25' and c['pair_id'] == claim['pair_id'] and
             c['initial_states'] == initials['ranks'] and c['native45'] == domain['native45'] and
             c['training800'] == domain['training800'] and c['occurrences'] == domain['occurrences'], 'ACTUAL_NATIVE_CONFIG_BINDING')
        owner = c['supervisor']
        need(owner['PID'] == install['PID'] and owner['startticks'] == install['startticks'] and
             owner['uid'] == os.getuid() and owner['host'] == install['host'] and owner['boot_id'] == install['boot_id'] and
             owner['scope_request_id'] == original['request_id'], 'SUPERVISOR_EXEC_IDENTITY')
        child = obj('run/child-owner.json')
        need(child['supervisor'] == owner and child['request_id'] == original['request_id'] and
             child['child']['request_id'] == original['request_id'] and child['child']['host'] == owner['host'] and
             child['child']['boot_id'] == owner['boot_id'] and child['child']['uid'] == os.getuid(), 'CHILD_HELD_IDENTITY')
        result['original_common_entry_terminal'] = absent(child['child'])
        resources = out['resources']
        result['RSS_sampled_peak_bytes'] = resources['sampled_group_peak_RSS_bytes']
        result['RSS_unknown_coverage_samples'] = resources.get('RSS_unknown_coverage_samples', 0)
        result['RSS_is_hard_limit'] = False
        result['RSS_is_observed_lower_bound'] = True
        need(out['native_returncode'] == 0 and out['failure'] is None and out['timed_out'] is False and
             out['release_permitted'] is True and len(out['rank_receipts']) == 8 and out['rank_receipt_errors'] == {}, 'TRUE_COMPLETE_C0')
        clean = out['cleanup']
        need(clean['held_child']['reaped'] is True and clean['termination_needed'] is False and
             clean['owned_processes_remaining'] == 0 and 'tree_error_type' not in clean, 'COMPLETE_HELD_TREE_CLEANUP')
        need(resources['soft_limit_bytes'] == 256 * 1024**3 and resources['hard_limit'] is False and
             resources['soft_limit_exceeded'] is False and resources['samples'] > 0 and
             0 < resources['sampled_group_peak_RSS_bytes'] <= 256 * 1024**3, 'QUALIFIED_RSS_SOFT_NOT_HARD')
        post = obj('run/after.json')
        need(post == out['post'] and post['active_compute_processes'] == 0 and post['master_port_bindable'] is True and
             len(post['devices']) == 8 and original['epoch'] <= post['observed_epoch'] <= out['finished_epoch'] <= original['epoch'] + 900,
             'ORIGINAL_BOUNDED8GPU_POST')
        expected_devices = {v['index']: v for v in c['expected_GPU_inventory']}
        need(len(expected_devices) == 8 and len({v['UUID'] for v in post['devices']}) == 8, 'UNIQUE_POST8_UUID')
        for device in post['devices']:
            expected = expected_devices[device['index']]
            need(device['UUID'] == expected['UUID'] and pci(device['PCI']) == pci(expected['PCI']) and
                 device['total_MiB'] == 40960 and 38000 <= device['free_MiB'] <= device['total_MiB'], 'ACTUAL_POST_UUID_PCI_MEMORY')
        result['resource_post_observed_clean'] = True
        entry_raw = load('run/common-entry-receipt.jsonl'); entry = [json.loads(line) for line in entry_raw.splitlines()]
        need([r['stage'] for r in entry] == ['planned', 'owned-cleanup-contract', 'fresh-admission', 'runtime', 'running', 'exited'] and
             entry[-1]['returncode'] == 0 and entry[0]['framework_members'] == 404 and
             entry[0]['launch']['environment_overrides']['MASTER_PORT'] == str(plan['master_ports']['generic']) and
             entry[1]['actual']['held_session_identity_verified'] is True and entry[1]['actual']['cleanup_helpers_verified'] is True,
             'SIX_ACTUAL_ENTRY_STAGES')
        actual_counts = Counter(); value_digests = []; rank_summary = []; totals = Counter()
        for rank in range(8):
            prefix = 'run/witness/rank-%02d/' % rank
            receipt = obj(prefix + 'receipt.json'); progress = json.loads(load(prefix + 'progress.json'))
            rank_owner = obj(prefix + 'owner.json')
            need(rank_owner['rank'] == rank and rank_owner['request_id'] == original['request_id'] and
                 rank_owner['epoch'] == original['epoch'] and rank_owner['deadline'] == original['epoch'] + 820 and
                 rank_owner['pair_id'] == claim['pair_id'] and rank_owner['arm'] == 'generic' and
                 rank_owner['mode'] == 'fresh25', 'ACTUAL_RANK_OWNER_CONTEXT')
            rank_terminal = absent(rank_owner)
            need(out['rank_receipts'][str(rank)]['file'] == metadata_pin(prefix + 'receipt.json'),
                 'CONTROLLER_RECEIPT_ACTUAL_FILE_PIN')
            initial = obj(prefix + 'initial-state.json'); live = obj(prefix + 'live25-values.json')
            rng = obj(prefix + 'saved25-rank-RNG.json'); micro = obj(prefix + 'microbatches.json')
            global800 = obj(prefix + 'global800-occurrences.json')
            need(receipt['rank'] == rank and receipt['mode'] == 'fresh25' and receipt['arm'] == 'generic' and
                 receipt['pair_id'] == claim['pair_id'] and receipt['completed_microbatches'] == 100 and receipt['updates'] == 25 and
                 receipt['completed_step'] == 25 and receipt['recovery_exact'] is None and receipt['formal17'] is False and
                 receipt['performance_qualified'] is False and receipt['effect_qualified'] is False, 'ALL8_REAL_RECEIPTS')
            need(progress['completed_microbatches'] == progress['planned_microbatches'] == 100 and
                 progress['completed_updates'] == list(range(1, 26)) and progress['saved_steps'] == [10, 20, 25] and
                 progress['error'] is None and progress['finished'] is True, 'ALL8_PROGRESS10025')
            need(initial['rank'] == rank and initial['named_SHA256'] == INITIAL_NAMED and initial['tensors'] == 504 and
                 initial['elements'] == 43646976 and initial['original_initial'] == initials['ranks'][str(rank)] and
                 initial['RNG_restored'] is True and initial['restoration']['RNG_restored'] is True and
                 initial['restoration']['original_rank_RNG_fingerprint'] == initial['restoration']['actual_RNG_after'], 'ORIGINAL_RANK_INITIAL_VALUES_AND_RNG')
            need(live['rank'] == rank and live['global_step'] == 25 and live['named_parameter_tensors'] ==
                 live['optimizer_state_entries'] == 504 and live['native_optimizer_tensor_values'] == 1512 and live['all8_values_equal'] is True and
                 receipt['all8_live_state_values_equal'] is True and
                 receipt['live_state_values'] == {k: live[k] for k in ('adapter', 'optimizer_named', 'scheduler')} and
                 all(digest(live[k]) for k in ('adapter', 'optimizer_named', 'scheduler')), 'REAL_NAMED_OPTIMIZER_VALUES_ALL8')
            value_digests.append(tuple(live[k] for k in ('adapter', 'optimizer_named', 'scheduler')))
            need(rng['rank'] == rank and rng['global_step'] == 25 and rng['exact'] is True and
                 rng['saved_fingerprint'] == rng['live_fingerprint'] and
                 rng['native_cuda_RNG_schema'] in ('current-device-tensor', 'eight-device-list'), 'ALL8_SAVED_RANK_RNG')
            inputs, forwards, losses = micro['training_inputs'], micro['forward_inputs'], micro['losses']
            need(micro['mode'] == 'fresh25' and micro['arm'] == 'generic' and micro['pair_id'] == claim['pair_id'] and
                 len(inputs) == len(forwards) == len(losses) == 100 and all(type(v) in (int, float) and math.isfinite(v) for v in losses),
                 'REAL_NATIVE100_FORWARD_BACKWARD_RETURNS')
            for j, (row, forward) in enumerate(zip(inputs, forwards)):
                need(row['microbatch'] == forward['microbatch'] == j and row['global_step_before'] == j // 4 and
                     type(row['canonical_ordinal']) is int and 0 <= row['canonical_ordinal'] < 45 and
                     type(row['input_tokens']) is int and 0 < row['input_tokens'] <= 4096 and
                     type(row['effective_supervised_tokens']) is int and 0 < row['effective_supervised_tokens'] < row['input_tokens'] and
                     row['supervised_EOS_count'] > 0 and digest(row['source_id_SHA256']) and digest(row['training_inputs_SHA256']) and
                     digest(row['collated_fields_SHA256']) and digest(forward['input_SHA256']) and forward['RNG_before'] is not None,
                     'ACTUAL_MICRO_INPUT_LOSS_TOKEN_IDENTITY')
                actual_counts[row['canonical_ordinal']] += 1
            input_tokens = sum(v['input_tokens'] for v in inputs); supervised = sum(v['effective_supervised_tokens'] for v in inputs)
            need(input_tokens == micro['actual_input_tokens'] and supervised == micro['actual_supervised_tokens'] and
                 global800['total'] == 800 and global800['counts'] == [Counter(indices)[i] for i in range(45)], 'ACTUAL_TOKEN800_COUNTS')
            for step in (1, 2):
                delta = obj(prefix + 'first-update-%d-actual-delta.json' % step)
                need(delta['global_step'] == delta['actual_optimizer_step'] == step and delta['optimizer_state_entries'] == 504 and
                     delta['parameter_delta']['parameter_tensors'] == 504 and
                     ((delta['parameter_delta']['changed_elements'] == 0 and all(v == 0 for v in delta['learning_rates']))
                      if step == 1 else delta['parameter_delta']['changed_elements'] > 0), 'REAL_FIRST2_NATIVE_UPDATES')
            totals['input_tokens'] += input_tokens; totals['supervised_tokens'] += supervised
            rank_summary.append(dict(rank=rank, microbatches=100, updates=25, input_tokens=input_tokens,
                supervised_tokens=supervised, first_loss=losses[0], final_loss=losses[-1],
                original_rank_terminal=rank_terminal,
                receipt_SHA256=pins[prefix + 'receipt.json']['SHA256'], microbatch_metadata_SHA256=pins[prefix + 'microbatches.json']['SHA256']))
            complete += 1
        need(len(set(value_digests)) == 1 and actual_counts == Counter(indices) and sum(actual_counts.values()) == 800,
             'ALL8_NAMED_VALUES_AND_ACTUAL800_CLOSURE')
        checkpoints = {}
        for step in (10, 20, 25):
            name = 'run/witness/rank-00/checkpoint-%d-files.json' % step; cp = obj(name)
            need(cp['path'] == str(SCOPE / 'run/train' / ('checkpoint-%d' % step)) and set(cp['files']) == MEMBERS and
                 all(type(v['bytes']) is int and 0 < v['bytes'] <= 512 * 1024**2 and digest(v['SHA256']) for v in cp['files'].values()) and
                 sum(v['bytes'] for v in cp['files'].values()) <= 512 * 1024**2, 'REAL_FULL14_CHECKPOINT_PINS')
            checkpoints[str(step)] = dict(manifest=metadata_pin(name), members=14,
                                         aggregate_bytes=sum(v['bytes'] for v in cp['files'].values()))
        named = obj('run/witness/rank-00/saved25-named-optimizer-values.json')
        need(named['rank'] == 0 and named['global_step'] == 25 and named['named_parameters'] == 504 and
             named['full_native_moment_tensors'] == 1512 and named['exact'] is True and
             named['saved_named_optimizer_SHA256'] == named['live_named_optimizer_SHA256'] == value_digests[0][1],
             'SAVED_TYPED_NAMED_OPTIMIZER_ACTUAL_VALUES')
        result.update(state='GENERIC25_FULL8_NATIVE_UPDATES_INDEPENDENT_METADATA_PASS', independent_passed=True,
            engineering_completed=True, release_permitted=True, completed_ranks=8, completed_updates=25, global_micro_exposures=800,
            all8_initial_named_SHA256=INITIAL_NAMED, all8_original_rank_RNG_restored=True,
            all8_named_optimizer_values_equal=True, final_live_values=dict(zip(('adapter', 'optimizer_named', 'scheduler'), value_digests[0])),
            saved25_named_optimizer_typed_comparison=True, all8_saved25_RNG_exact=True,
            actual_input_tokens=totals['input_tokens'], actual_shifted_supervised_tokens=totals['supervised_tokens'],
            actual_unique45_occurrence_counts=[actual_counts[i] for i in range(45)], ranks=rank_summary, checkpoints=checkpoints,
            owned_cleanup_observed=True, all_UID_FDs_inspected=False, exclusive_GPU_ownership_proven=False,
            original_T03_recovery_drift_retained=True, original_T04_failure_not_reclassified=True)
        return finish()
    except BaseException as error:
        result.update(state='INDEPENDENT_METADATA_INCOMPLETE_OR_UNKNOWN', failure_type=type(error).__name__,
            failure_code=str(error)[:160] if isinstance(error, ValueError) else 'BOUNDED_METADATA_READ_OR_SCHEMA_ERROR',
            completed_ranks=complete, observed_rank_progress=observed_ranks, producer_failure_not_reclassified=True)
        return finish()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

if __name__ == '__main__':
    result = read_terminal(READ_CLAIM, READ_ORIGIN, READ_REQUEST)
    print(canonical(result).decode(), end='')
    raise SystemExit(0 if result['independent_passed'] else 1)
