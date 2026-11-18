"""Offline consume the exact v2 11-small-original bundle; never query or launch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REQUEST = '80b3f2c3305340b38913c2080a4cae5d'
PUBLICATION = '9629d5d62f70e78a6181be8d4c56d4f857d9f615'
MF = 'aa2f94b66943a553073f8fda00b63c6b864ecfaea138ab7064f5ab2750c4f454'
CONTEXT = {
    'bundle': dict(bytes=6109, SHA256='ecfc03866e1a12bd8438810db03dc0c5c35e8b94e5b3c697c48fa862edbdbe9d'),
    'intent': dict(bytes=603, SHA256='bf92c5588ed778dd9a7ccfc5d0058cf5a6ab4c6abd2f4192c31136da15a85fa1'),
    'receipt': dict(bytes=969, SHA256='2d92e627e27c8269a85b1a4fa127a687544e94e12fd15d9094799f103a3fd157'),
}
PRESENT = {
    'run/outcome.json': (1919, 'b63395e3984d6b071b544ec833e8f2cf0de55c276916f566494c5204fcbd073b'),
    'run/failure-private.json': (1186, '739e978525440abfbc1df9fe72fd5e01379daee51cc0771f0738f03b92ef9b6c'),
    'run/control-state.json': (604, 'f9e1ae4e04ca3e4444016950c349c255fedd4107e1bf9afbae0099adb9465f3d'),
    'install-intent.json': (575, 'f01278767faeaaec8b4f18b4ffa439b9a8a6670ff9143e8d6d5878b3263e1fc1'),
    'run/resource-samples.json': (291, 'e4c749cf8bc4689ccbf12600af2fcf7658e1892cece47df466090ffe2a4aee0e'),
}
MISSING = {'run/admission-query-failure-private.json', 'run/post-query-failure-private.json',
           'run/child-owner.json', 'run/after.json', 'run/before.json', 'install-failure.json'}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read(path, expected):
    before = path.lstat(); need(path.is_file() and not path.is_symlink() and before.st_size == expected['bytes'], 'EXACT_FILE_BOUND')
    body = path.read_bytes(); after = path.lstat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
         and sha(body) == expected['SHA256'], 'ACTUAL_LOCAL_ORIGINAL_SHA')
    return json.loads(body)


def consume(a):
    bundle = read(a.bundle, CONTEXT['bundle']); intent = read(a.dispatch / 'intent.json', CONTEXT['intent'])
    receipt = read(a.dispatch / 'receipt.json', CONTEXT['receipt'])
    need(bundle['request_id'] == REQUEST and bundle['read_only'] is True
         and bundle['prediction_gold_or_weight_body_reads'] == bundle['device_queries'] == 0, 'PASSIVE_BUNDLE_CONTEXT')
    need(set(bundle['files']) == set(PRESENT) | MISSING and len(bundle['files']) == 11, 'EXACT_ELEVEN_SMALL_SLOTS')
    for name, (size, digest) in PRESENT.items():
        row = bundle['files'][name]; need(row['bytes'] == size and row['SHA256'] == digest and isinstance(row['value'], dict), 'ORIGINAL_REMOTE_FILE_PIN')
    need(all(bundle['files'][name] == {'state':'MISSING'} for name in MISSING), 'MISSING_STATUS_PRESERVED')
    get = lambda name: bundle['files'][name]['value']
    outcome = get('run/outcome.json'); control = get('run/control-state.json'); install = get('install-intent.json')
    failure = get('run/failure-private.json'); resources = get('run/resource-samples.json')
    for value in [intent, receipt, outcome, control, install]:
        need(value['request_id'] == REQUEST and value['publication'] == PUBLICATION and value['source_manifest_SHA256'] == MF,
             'ONE_ACTUAL_REQUEST_PUBLICATION_SOURCE')
        need(value['epoch'] == 1794961637.35619, 'ORIGINAL_EPOCH')
    need(intent['exec_request_attempted'] is False and receipt['exec_request_attempted'] is True
         and receipt['remote_outcome'] == 'KNOWN_TERMINAL' and receipt['rc'] == 1
         and receipt['replay_allowed'] is False, 'SSH_ATTEMPT_IS_SEPARATE_FROM_NATIVE_ATTEMPT')
    need(outcome['engineering_completed'] is False and outcome['release_permitted'] is False
         and outcome['native_released'] is False and outcome['native_returncode'] is None
         and control['exec_request_attempted'] is control['entry_released'] is False,
         'NO_NATIVE_START_BY_SAVED_CONTROL_FLOW')
    need(outcome['failure']['phase'] == failure['phase'] == 'passive_admission'
         and failure['message'] == 'LIVE_OWN_UID_FD_INVENTORY_UNKNOWN'
         and sha(failure['message'].encode()) == outcome['failure']['message_SHA256'], 'ACTUAL_SAVED_FD_VISIBILITY_FAILURE')
    need(outcome['failure']['private_file']['SHA256'] == PRESENT['run/failure-private.json'][1]
         and outcome['post'] == {'state':'POST_UNKNOWN','error_type':'ValueError'}, 'FAILED_POST_NOT_DEVICE_CLEAN')
    owner = control['supervisor']
    need(owner['host'] == bundle['host'] == install['host'] and owner['boot_id'] == bundle['boot_id'] == install['boot_id']
         and owner['PID'] == install['PID'] == 1365233 and owner['startticks'] == install['startticks'] == 738172500
         and owner['uid'] == install['UID'] and owner['scope_request_id'] == REQUEST, 'SAVED_STRONG_CONTROLLER_IDENTITY')
    process, = bundle['process_observations']
    need(process['original'] == owner and process['state'] == 'ABSENT', 'SAME_HOST_BOOT_SAVED_OWNER_ABSENT')
    need(resources['samples'] == 0 and resources['last_sample'] is None and resources['inherited_affinity'] == [2,3,4,5], 'UNSAMPLED_RSS_IS_NOT_MEASURED_ZERO')
    need(outcome['planned_generations'] == 30 and outcome['actual_training_updates'] == 0
         and outcome['labels_read'] is outcome['blind_opened'] is outcome['original_domain_training_accepted'] is False,
         'GOLD_CLOSED_AND_ORIGINAL_FAILURE_RETAINED')
    # Review the exact locally published source, not a new collector or target import.
    mf_raw = (a.source / 'SOURCE_FILES.json').read_bytes(); need(sha(mf_raw) == MF, 'ACTUAL_PUBLISHED_V2_SOURCE')
    members = json.loads(mf_raw)['files']; bodies = {}
    for name in ['bounded_eval.py', 'gpu_admission.py']:
        body = (a.source / name).read_bytes(); need(len(body) == members[name]['bytes'] and sha(body) == members[name]['SHA256'], 'PUBLISHED_CAUSAL_SOURCE')
        bodies[name] = body
    tree = ast.parse(bodies['bounded_eval.py'])
    gpu_lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'gpu_snapshot']
    popen_lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen']
    need(min(gpu_lines) < min(popen_lines) and 'selected_fd_holders' in failure['trace'], 'PRE_CHILD_FAILURE_SOURCE_ORDER')
    gpu_source = bodies['gpu_admission.py'].decode()
    need("devices = query('index,uuid,pci.bus_id,memory.free,memory.total'" in gpu_source
         and gpu_source.index('information = driver_information') < gpu_source.index('fds = selected_fd_holders'), 'SUPPORTED5_AND_MINOR_MAPPING_PRECEDE_FD_GUARD')
    return dict(state='ORIGINAL_V2_KNOWN_TERMINAL_FD_VISIBILITY_ADMISSION_FAILURE', independent_failure_metadata_verified=True,
        request_id=REQUEST, publication=PUBLICATION, source_manifest_SHA256=MF,
        original_client_returncode=1, original_failure_phase='passive_admission',
        original_failure_constant=failure['message'], ordinary_SSH_request_attempted=True,
        native_exec_request_attempted=False, native_released=False, native_child_not_started_by_control_and_frozen_source=True,
        saved_controller=dict(PID=owner['PID'],startticks=owner['startticks'],same_host_boot_UID_verified=True,observed_state='ABSENT'),
        original_engineering_completed=False, original_release_permitted=False,
        independent_device_post='UNKNOWN', selected_GPU_FD_visibility='UNKNOWN_UNREADABLE_OWN_UID_PROC_ITEMS',
        exact_unreadable_PID_list='NOT_SAVED_BY_ORIGINAL_V2_GUARD; separate root diagnostic pending',
        supported_query_and_minor_stage='Passed to FD guard by original failure trace and exact source order; no completed admission snapshot',
        before_after_device_receipts_complete=False, measured_GPU_busy=None, observed_peak_RSS_bytes=None,
        producer_cleanup_scope='No child held; original outcome says no owned descendants. Not a complete device post.',
        planned_generations=30, observed_predictions=0, missing_predictions=30,
        arms={arm:dict(denominator=10,observed=0,missing=10,correct=None,accuracy=None,scoring='NOT_RUN_GOLD_CLOSED') for arm in ['initial','domain25','generic25']},
        dev_gold_read=False, blind_opened=False, actual_training_updates=0, original_domain_failure_preserved=True,
        replay_allowed=False, effect_qualified=False, formal17=False, GPU_GO=False, Model_GO=False,
        original_bundle=CONTEXT['bundle'], local_intent=CONTEXT['intent'], local_receipt=CONTEXT['receipt'],
        original_small_slots=11,present_small_slots=5,missing_small_slots=6,
        independent_review_boundary='Remote root first-reader pinned exact original bytes and strong owner state; this offline reader verifies bundle/source/context and parsed values, not reconstructed raw nested JSON hashes.',
        new_reviewer_remote_calls=0,new_reviewer_device_queries=0,new_reviewer_prediction_gold_or_weight_body_reads=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['bundle','dispatch','source','output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); value = consume(a)
    with a.output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k:value[k] for k in ['state','request_id','independent_failure_metadata_verified','observed_predictions','missing_predictions','independent_device_post']},sort_keys=True))
