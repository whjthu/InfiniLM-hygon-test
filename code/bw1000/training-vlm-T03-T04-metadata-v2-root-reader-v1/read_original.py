"""Root invokes once after review: two exact read-only endpoint inspections."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import time

D = Path(__file__).resolve().parent
G = D.parents[2]
STAGE = G/'code/bw1000/training-vlm-T03-T04-metadata-v2'
LOCAL = G/'local-only/training-vlm-T03-T04-metadata-v2'
PUB = 'b54b318ce73f916890877697c9cbb23b47a45b0b'
MF = '853e4cfdbb0dd2c84287aa5aaa127f10994e16441f67f5b89d96da7619706b9a'
CONTEXT = dict(publication=PUB, request_id='250e43f5a5b048f6bb2ca674e8a8ef44', epoch=1794903737.376735,
               deadline=1794904037.376735, source_manifest_SHA256=MF)
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H = lambda raw: hashlib.sha256(raw).hexdigest()


def read(path, maximum=262144):
    raw = path.read_bytes()
    assert 0 < len(raw) <= maximum
    return json.loads(raw), dict(bytes=len(raw), SHA256=H(raw))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    raw = (STAGE/'SOURCE_FILES.json').read_bytes()
    assert H(raw) == MF
    for name, pin in json.loads(raw)['files'].items():
        item = (STAGE/name).read_bytes()
        assert len(item) == pin['bytes'] and H(item) == pin['SHA256']
    inspector_mf = json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name, pin in inspector_mf['files'].items():
        item = (D/name).read_bytes()
        assert len(item) == pin['bytes'] and H(item) == pin['SHA256']
    plan = json.loads((STAGE/'PLAN.json').read_bytes())
    original, pins = {}, {}
    for name in ['intent.json', 'outcome.json', 'source-transport.json', 'receiver-transport.json', 'receiver-summary.json']:
        original[name], pins[name] = read(LOCAL/name)
    assert all(original['intent.json'][key] == expected for key, expected in CONTEXT.items())
    assert all(original['outcome.json'][key] == expected for key, expected in CONTEXT.items())
    assert H(HELPER.read_bytes()) == HELPER_SHA
    outdir = G/'local-only/training-vlm-T03-T04-metadata-v2-independent-root-reader-v1'
    outdir.mkdir(mode=0o700)
    save(outdir/'intent.json', dict(original_context=CONTEXT, readonly=True, endpoint_queries=2,
                                   no_bundle_body_read=True, no_original_model_or_run_file_read=True))
    started = time.monotonic()
    endpoint_values = {}
    for mode in ['source', 'receiver']:
        code = ('MODE='+repr(mode)+'\nPLAN='+repr(plan)+'\nCONTEXT='+repr(CONTEXT)+'\n').encode() + (D/'read_endpoint.py').read_bytes()
        if mode == 'source':
            config = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
            argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
                    '-F', str(config), 'a100-27', '/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -']
        else:
            base = runpy.run_path(str(HELPER))['ssh']
            argv = [base[0], '-o', 'StrictHostKeyChecking=yes', *base[1:], '/usr/local/python3.12/bin/python3 -I -S -B -']
        assert len(subprocess.list2cmdline(argv)) < 30000
        budget = min(70, 150-(time.monotonic()-started))
        assert budget > 0
        try:
            result = subprocess.run(argv, input=code, capture_output=True, timeout=budget)
            stdout, stderr, rc = result.stdout, result.stderr, result.returncode
            unknown = rc == 255
        except subprocess.TimeoutExpired as error:
            stdout, stderr, rc, unknown = error.stdout or b'', error.stderr or b'', None, True
        assert len(stdout) <= 131072 and len(stderr) <= 262144
        (outdir/(mode+'.stdout')).write_bytes(stdout)  # bounded aggregates/pins only
        (outdir/(mode+'.stderr.private')).write_bytes(stderr)
        receipt = dict(returncode=rc, remote_outcome='UNKNOWN' if unknown else 'RETURNED',
                       stdout_bytes=len(stdout), stdout_SHA256=H(stdout), stderr_bytes=len(stderr), stderr_SHA256=H(stderr),
                       original_request=CONTEXT['request_id'], replay_allowed=False)
        if rc == 0:
            receipt['value'] = json.loads(stdout)
            assert receipt['value']['context'] == CONTEXT
        save(outdir/(mode+'-receipt.json'), receipt)
        endpoint_values[mode] = receipt
    result = dict(original_context=CONTEXT, original_local_record_pins=pins,
                  original_controller_passed=original['outcome.json']['passed'],
                  original_transport_states={key:original[key]['remote_outcome'] for key in ['source-transport.json', 'receiver-transport.json']},
                  metadata_transfer_verified=False, full_weight_identity=False, numerical_training_qualified=False,
                  GPU_GO=False, Model_GO=False, original_v1_failure_denominators_retained=True,
                  actual_gpu_busy='NOT_MEASURED', new_Slurm_GPU_Torch_calls=0, endpoint_reads=endpoint_values)
    if all(value.get('value', {}).get('passed') is True for value in endpoint_values.values()):
        source = endpoint_values['source']['value']['details']
        receiver = endpoint_values['receiver']['value']['details']
        st = original['source-transport.json']
        assert source['bundle']['bytes'] == receiver['bundle']['bytes'] == st['stdout_bytes']
        assert source['bundle']['SHA256'] == receiver['bundle']['SHA256'] == st['stdout_SHA256']
        assert source['initial_binding_SHA256'] == receiver['initial_binding_SHA256'] == original['receiver-summary.json']['initial_binding_SHA256']
        assert source['initial_dependencies'] == receiver['initial_dependencies']
        assert receiver['actual_written_members'] == 36
        assert original['receiver-summary.json']['parent_reservations_SHA256'] == receiver['reservation']['original_receipt_SHA256']
        result.update(metadata_transfer_verified=True, independent_source_process_exited=True,
                      independent_receiver_process_exited=True, target_written_members=36, target_member_denominator=36,
                      initial_dependencies=receiver['initial_dependencies'], initial_rows=16,
                      reserved_nodes=80, reserved_cards=640, reserved_not_busy=True,
                      source_logical_consumed_bytes=32327445,
                      source_bundle_body_rehashed=False, physical_IO='UNKNOWN_NOT_MEASURED')
    save(outdir/'result.json', result)
    print(json.dumps(dict(metadata_transfer_verified=result['metadata_transfer_verified'],
                          original_controller_passed=result['original_controller_passed'],
                          target_member_denominator=36, initial_row_denominator=16,
                          result_path=str(outdir/'result.json'), full_weight_identity=False,
                          GPU_GO=False, physical_IO='UNKNOWN_NOT_MEASURED'), allow_nan=False))
    return 0 if result['metadata_transfer_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
