"""Install this exact small CPU package and train/dev inputs, then exec its owned supervisor."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

A = Path('/home/whj/nvidia-baseline/alignment-assets')
PY = '/home/whj/nvidia-baseline/env-vlm/bin/python'
OLD = A / 'hygon-vlm-cuda-resume20-v1-package'
FRAMEWORK = A / 'training-USGS-T03-fresh-domain-A100-v1/framework'
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
    need(isinstance(INSTALL_PAYLOAD_B64, str) and len(INSTALL_PAYLOAD_B64) < 6 * 1024**2,
         'bounded root payload required')
    payload_raw = base64.b64decode(INSTALL_PAYLOAD_B64, validate=True)
    need(len(payload_raw) <= 4 * 1024**2, 'root payload exceeded')
    payload = json.loads(payload_raw)
    case = payload['case']
    need(case in MODELS, 'exact CPU case required')
    epoch = payload['epoch']
    need(type(epoch) in (int,float) and math.isfinite(epoch) and 0 < epoch <= time.time() < epoch+290, 'original epoch exhausted')
    mono = time.monotonic()
    original_remaining = epoch + 290 - time.time()

    def check():
        need(min(epoch + 290 - time.time(), original_remaining - (time.monotonic() - mono)) > 0,
             'original CPU installation epoch exhausted')

    def expired(*unused):
        raise TimeoutError('original CPU installation epoch exhausted')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, original_remaining)
    scope = A / 'training-USGS-T03-wide150-native-CPU-v1'
    need(scope.parent == A and not scope.exists() and not scope.is_symlink(), 'original namespace exists; reconcile')
    scope.mkdir(mode=0o700)
    try:
        write(scope / 'install-intent.json', (json.dumps(dict(
            case=case, request_id=payload['request_id'], publication=payload['publication'],
            epoch=epoch, deadline=epoch + 300, planned_native_rows=150, source_manifest_SHA256=payload['source_manifest_SHA256'],
            claim_SHA256=payload['claim_SHA256'], host=os.uname().nodename, UID=os.getuid(), PID=os.getpid(), startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
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
        plan = json.loads(package['PLAN.json'])
        binding = plan['cases'][case]
        need(binding['model_metadata']['model_root'] == MODELS[case], 'model metadata root changed')
        need(str(FRAMEWORK)==plan['framework'],'ACTUAL404_FRAMEWORK_PATH')
        allowed=sorted(set(os.sched_getaffinity(0))-{0,1});need(len(allowed)>=2,'EXACT2_CPU_ALLOWED')
        os.sched_setaffinity(0,set(allowed[:2]))
        data={name:base64.b64decode(raw,validate=True) for name,raw in payload['data'].items()}
        db_raw=data['data-binding-private.json'];need(digest(db_raw)==payload['data_binding_SHA256'],'PRIVATE_DATA_BINDING_PIN')
        db=json.loads(db_raw)
        need(db['target_data']==str(scope/'data') and db['planned']==plan['planned'] and db['media']==plan['media']
             and db['source_inputs']==plan['source_inputs'] and db['dataset_manifest_SHA256']==plan['dataset_manifest']['SHA256']
             and db['dataset_audit_SHA256']==plan['dataset_audit']['SHA256'] and db['dev_gold_read'] is False
             and db['blind_gold_read'] is False and db['blind_inputs_and_media_read'] is True,'DATA_SOURCE_AND_PRIVACY_BINDING')
        expected=set(db['files'])|set(db['media'])|{'data-binding-private.json'}
        need(set(data)==expected and len(db['media'])==30 and set(db['files'])=={x+'-swift.jsonl' for x in plan['planned']},'EXACT34_DATA_MEMBERS')
        for name,body in data.items():
            check();part=__import__('pathlib').PurePosixPath(name)
            need(not part.is_absolute() and '..' not in part.parts and '\\' not in name,'DATA_MEMBER_PATH')
            if name!='data-binding-private.json':
                pin=db['files'].get(name,db['media'].get(name));need(len(body)==pin['bytes'] and digest(body)==pin['SHA256'],'DATA_MEMBER_PIN')
            write(scope/'data'/name,body)
        for split in plan['planned']:
            need(db['files'][split+'-swift.jsonl']['rows']==50,'ALL150_INPUTS')
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
                   PYTHONNOUSERSITE='1', PYTHONPATH=str(scope / 'src') + ':' + str(FRAMEWORK))
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
                      state='KNOWN_INSTALLATION_FAILURE_NO_REPLAY', planned_native_rows=150, error_type=type(error).__name__,
                      error_message_SHA256=digest(str(error).encode()), GPU_calls=0, training_updates=0)
        write(scope / 'install-failure.json', (json.dumps(record) + '\n').encode())
        print(json.dumps(record), flush=True)
        raise


if __name__ == '__main__':
    main()
