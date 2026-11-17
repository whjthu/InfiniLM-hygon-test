"""Install this exact small CPU package and train/dev inputs, then exec its owned supervisor."""
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import time

A = Path('/home/whj/nvidia-baseline/alignment-assets')
PY = '/home/whj/nvidia-baseline/env-vlm/bin/python'
OLD = A / 'hygon-vlm-cuda-resume20-v1-package'
FRAMEWORK = OLD / 'framework'
MODELS = {
    'T03.sft': '/mnt/data/nvidia-baseline-20261010/inputs/models/InternVL3_5-8B-HF',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    need(not path.exists() and not path.is_symlink(), 'new member already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def pinned(path, pin, cap):
    need(path.is_file() and not path.is_symlink() and 0 < pin['bytes'] <= cap, 'bounded source absent')
    before = path.stat()
    need(before.st_size == pin['bytes'], 'source size differs')
    with path.open('rb') as stream:
        raw = stream.read(pin['bytes'])
        after = os.fstat(stream.fileno())
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and
         len(raw) == pin['bytes'] and digest(raw) == pin['SHA256'], 'source identity differs')
    return raw


def main():
    need(isinstance(INSTALL_PAYLOAD_B64, str) and len(INSTALL_PAYLOAD_B64) < 3 * 1024**2,
         'bounded root payload required')
    payload_raw = base64.b64decode(INSTALL_PAYLOAD_B64, validate=True)
    need(len(payload_raw) < 2 * 1024**2, 'root payload exceeded')
    payload = json.loads(payload_raw)
    case = payload['case']
    need(case in MODELS, 'exact CPU case required')
    epoch = payload['epoch']
    need(0 < epoch + 290 - time.time() <= 290, 'original epoch exhausted')
    mono = time.monotonic()
    original_remaining = epoch + 290 - time.time()

    def check():
        need(min(epoch + 290 - time.time(), original_remaining - (time.monotonic() - mono)) > 0,
             'original CPU installation epoch exhausted')

    def expired(*unused):
        raise TimeoutError('original CPU installation epoch exhausted')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, original_remaining)
    scope = A / 'training-USGS-T03-native-template-CPU-v2'
    need(scope.parent == A and not scope.exists() and not scope.is_symlink(), 'original namespace exists; reconcile')
    scope.mkdir(mode=0o700)
    try:
        write(scope / 'install-intent.json', (json.dumps(dict(
            case=case, request_id=payload['request_id'], publication=payload['publication'],
            epoch=epoch, deadline=epoch + 300, source_manifest_SHA256=payload['source_manifest_SHA256'],
            claim_SHA256=payload['claim_SHA256'], host=os.uname().nodename, UID=os.getuid(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())) + '\n').encode())
        package = {name: base64.b64decode(encoded, validate=True)
                   for name, encoded in payload['package'].items()}
        manifest_raw = package['SOURCE_FILES.json']
        need(digest(manifest_raw) == payload['source_manifest_SHA256'], 'package manifest differs')
        manifest = json.loads(manifest_raw)['files']
        need(set(package) == set(manifest) | {'SOURCE_FILES.json'}, 'exact package members required')
        for name, raw in package.items():
            check()
            need(Path(name).name == name and name not in {'.', '..'}, 'package path escape')
            if name != 'SOURCE_FILES.json':
                need(len(raw) == manifest[name]['bytes'] and digest(raw) == manifest[name]['SHA256'],
                     'package member differs')
            write(scope / 'src' / name, raw)
        claim = json.loads(package['CLAIM.json'])
        need(digest(package['CLAIM.json']) == payload['claim_SHA256'] and
             claim['root_GO'] is True and claim['CPU_GO'] is True and
             claim['GPU_GO'] is False and claim['training_GO'] is False, 'CPU scope not adopted')
        prior_raw = package['PRIOR_V1_FAILURE.json']
        prior = json.loads(prior_raw)
        need(digest(prior_raw) == claim['prior_v1_failure']['SHA256'] and
             prior['request_id'] == '1e0a5eb9fac94bcd8a5f2387908a3545' and prior['worker_returncode'] == 1 and
             prior['independent_postclean'] is True and prior['actual_completed_train'] == 0 and
             prior['actual_completed_dev'] == 0 and prior['failed_or_uncompleted_rows_retained'] == 55,
             'original v1 terminal failure is not independently closed')
        plan = json.loads(package['PLAN.json'])
        binding = plan['cases'][case]
        need(binding['model_metadata']['model_root'] == MODELS[case], 'model metadata root changed')
        data = {name: base64.b64decode(encoded, validate=True)
                for name, encoded in payload['data'].items()}
        need(set(data) == set(plan['media']) | {
            'train-swift.jsonl', 'dev-swift.jsonl', 'data-binding-private.json'}, 'exact train/dev-only data required')
        db_raw = data['data-binding-private.json']
        db = json.loads(db_raw)
        need(db['passed'] is True and db['target_data'] == str(scope / 'data') and
             db['dataset_manifest_SHA256'] == plan['dataset_manifest']['SHA256'] and
             db['dataset_audit_SHA256'] == plan['dataset_audit']['SHA256'] and
             db['dev_gold_read'] is False and db['blind_inputs_gold_or_media_read'] is False,
             'adapter qualification differs')
        need(db['media'] == plan['media'], 'media declaration differs')
        pins = dict(plan['media'], **db['files'])
        for name, raw in data.items():
            check()
            parts = Path(name).parts
            need(not Path(name).is_absolute() and '..' not in parts and
                 'blind' not in parts, 'data path escape')
            if name != 'data-binding-private.json':
                need(len(raw) == pins[name]['bytes'] and digest(raw) == pins[name]['SHA256'],
                     'data member differs')
            write(scope / 'data' / name, raw)
        native_path = OLD / 'package/cases' / (case + '-native-binding.json')
        native_raw = pinned(native_path, binding['native_binding_pin'], 2 * 1024**2)
        write(scope / 'native-binding.json', native_raw)
        model = Path(MODELS[case])
        metadata = binding['model_metadata']['files']
        for name, pin in metadata.items():
            check()
            need(Path(name).name == name and not name.endswith(('.safetensors', '.pt', '.pth', '.bin', '.ckpt')),
                 'metadata-only exact member required')
            write(scope / 'model-metadata' / model.name / name, pinned(model / name, pin, 16 * 1024**2))
        allowed = sorted(set(os.sched_getaffinity(0)) - {0, 1})
        need(len(allowed) >= 2, 'two CPU cores required')
        os.sched_setaffinity(0, set(allowed[:2]))
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   WANDB_DISABLED='true', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(scope / 'src') + ':' + str(FRAMEWORK))
        check()
        need(time.time() < epoch + 240, 'no adequate native work reserve remains')
        (scope / 'run').mkdir(mode=0o700)
        command = [PY, '-B', '-X', 'utf8', str(scope / 'src/run_CPU_once.py'),
                   '--case', case, '--framework', str(FRAMEWORK),
                   '--native-binding', str(scope / 'native-binding.json'),
                   '--model-view', str(scope / 'model-metadata' / model.name), '--data', str(scope / 'data'),
                   '--data-binding-sha', digest(db_raw), '--data-binding-bytes', str(len(db_raw)),
                   '--scope', str(scope / 'run'), '--request-id', payload['request_id'],
                   '--publication', payload['publication'], '--epoch', str(epoch)]
        os.execve(PY, command, env)
    except BaseException as error:
        record = dict(case=case, request_id=payload['request_id'], epoch=epoch,
                      state='KNOWN_INSTALLATION_FAILURE_NO_REPLAY', error_type=type(error).__name__,
                      error=str(error)[:200], GPU_calls=0, training_updates=0)
        write(scope / 'install-failure.json', (json.dumps(record) + '\n').encode())
        print(json.dumps(record), flush=True)
        raise


if __name__ == '__main__':
    main()
