"""Install one exact public-data carrier and exercise the original VGGT CPU input path."""

from base64 import b64decode
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import socket
import stat
import subprocess
import sys
import tarfile
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
ORIGINAL = ROOT / 'incoming/vggt-3493-cp25-cpu-94b06a02507a/original-carrier.py'
ORIGINAL_SHA = 'dbb5e1bebc5463fe0830cf4c351a3ba052faf8dba0af8efa4b19fadd218f73c0'
SOURCE_SHA = '6b345869c2b45516f7271a29543d2d3481dc0d1e6e77c560444fecbe65e71fac'
INPUT_SHA = '01ed8f1ae762a3b2736aeb76ebed1cca73f518072ef777613ad2bcb59f0305eb'
STAGE, TRANSFER = '3ec7343a97bb41c097e903056a90852c', 'b07cf220f7d04a09afa1919e6869be9d'
CPU = '8e68f5d43e384bb1be937d38b9cff371'
LABEL = 'vggt-household-native-cpu-' + CPU[:12]
RUN = ROOT / 'incoming' / LABEL
SRC, ASSETS = RUN / 'source', RUN / 'assets'
PARENT, NODE = '1103606', 'f13r1n12'
MANAGED = '/usr/local/python3.12/bin/python3'
EXPECTED_BYTES = 24_200_249
TAR_CAP = 25 << 20
PNG = re.compile(r'(rgb|depth)/[0-9]+\.[0-9]+\.png')
PREFIX = b'VGGT_HOUSEHOLD_NATIVE_CPU '


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(data):
    return {'SHA256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def write(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return pin(data)


def sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path, cap):
    need(path.resolve(strict=True) == path, 'canonical_artifact')
    with path.open('rb') as stream:
        start = os.fstat(stream.fileno())
        data = stream.read(cap + 1)
        end = os.fstat(stream.fileno())
    current = path.lstat()
    key = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(stat.S_ISREG(current.st_mode) and key(start) == key(end) == key(current)
         and 0 < len(data) == current.st_size <= cap, 'stable_bounded_artifact')
    return data


def input_spec(raw):
    need(pin(raw)['SHA256'] == INPUT_SHA, 'exact_public_manifest')
    files = json.loads(raw)['files']
    meta = {'rgb.txt', 'depth.txt', 'groundtruth.txt'}
    need(len(files) == 79 and meta <= files.keys() and all(PNG.fullmatch(k) for k in files.keys() - meta)
         and sum(info['bytes'] for info in files.values()) == EXPECTED_BYTES, 'only_the_79_committed_objects')
    return files


def stage(data):
    need(data['request'] == STAGE and re.fullmatch('[a-f0-9]{40}', data['publication']), 'exact_source_stage')
    items = {name: b64decode(value, validate=True) for name, value in data['files'].items()}
    raw = items.pop('SOURCE_FILES.json')
    need(pin(raw)['SHA256'] == SOURCE_SHA and items.keys() == {'native_entry.py', 'tum_vggt_pair.py', 'INPUT_FILES.json'}
         and json.loads(raw)['files'] == {k: pin(v) for k, v in items.items()}, 'exact_three_source_objects')
    input_spec(items['INPUT_FILES.json'])
    carrier = b64decode(data['carrier'], validate=True)
    need(0 < len(carrier) <= 32768 and pin(carrier)['SHA256'] == data['carrier_SHA256'], 'exact_carrier')
    RUN.mkdir(mode=0o700)
    SRC.mkdir(mode=0o700)
    for name,value in {**items, 'SOURCE_FILES.json':raw}.items():
        write(SRC/name,value)
    sync(SRC)
    write(RUN/'carrier.py',carrier)
    receipt = {'request': STAGE, 'publication': data['publication'], 'source_SHA256': SOURCE_SHA,
               'carrier_SHA256': data['carrier_SHA256'], 'passed': True, 'actual_GPU': 0}
    install = write(RUN/'installed.json',encode(receipt))
    sync(RUN)
    return dict(receipt, installed=install)


def installation(expected):
    data = read(RUN/'installed.json',4096)
    need(pin(data)['SHA256'] == expected, 'exact_installed_receipt')
    item = json.loads(data)
    need(item['source_SHA256'] == SOURCE_SHA and item['request'] == STAGE
         and item['carrier_SHA256'] == pin(read(RUN/'carrier.py',32768))['SHA256'], 'exact_installed_identity')
    return item


def verify_tree(root, files):
    actual = set()
    for path in root.rglob('*'):
        need(not path.is_symlink(), 'no_aliases_in_staged_tree')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    need(actual == files.keys(), 'no_extra_or_missing_staged_file')
    return {name: pin(read(root/name, 4 << 20)) for name in files} == files


def unpack(tar, output, files):
    seen = set()
    for entry in tar:
        name = entry.name
        need(name in files and name not in seen and entry.isfile() and entry.size == files[name]['bytes'],
             'exact_manifest_names_and_extents')
        with tar.extractfile(entry) as source:
            value = source.read(entry.size+1)
        need(pin(value) == files[name], 'exact_manifest_member_digest')
        target = output/name
        target.parent.mkdir(mode=0o700, exist_ok=True)
        write(target, value)
        seen.add(name)
    need(seen == files.keys() and verify_tree(output, files), 'all_79_files_rehashed_after_write')


def transfer(args):
    request, nonce, install, tar_sha, count_text = args
    need(request == TRANSFER and all(re.fullmatch('[0-9a-f]{64}', v) for v in (install, tar_sha))
         and re.fullmatch('[0-9a-f]{32}', nonce), 'exact_public_stream_identity')
    count = int(count_text)
    need(EXPECTED_BYTES <= count <= TAR_CAP and count % 512 == 0, 'exact_bounded_tar_extent')
    installed = installation(install)
    need(not (RUN/'transfer-intent.json').exists() and not ASSETS.exists() and not (RUN/'assets.partial').exists(),
         'no_replay_or_existing_transfer')
    intent = {'request': request, 'nonce': nonce, 'publication':installed['publication'],
              'input_SHA256': INPUT_SHA, 'tar_SHA256':tar_sha, 'bytes':count, 'installed_SHA256': install}
    write(RUN/'transfer-intent.json',encode(intent));sync(RUN)
    print(json.dumps(dict(intent,event='BOUND')),flush=True)
    import signal
    def timeout(*_):
        raise TimeoutError('public_payload_budget')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    raw = sys.stdin.buffer.read(count+1)
    need(len(raw) == count and pin(raw)['SHA256'] == tar_sha, 'exact_transport_length_and_digest')
    partial = RUN/'assets.partial'
    partial.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
        unpack(archive, partial, input_spec(read(SRC/'INPUT_FILES.json',65536)))
    for d in (partial/'rgb',partial/'depth',partial):
        sync(d)
    partial.rename(ASSETS)
    sync(RUN)
    signal.alarm(0)
    answer = dict(intent, passed=True, verified_objects=79, verified_manifest_bytes=EXPECTED_BYTES, actual_GPU=0)
    receipt = write(RUN/'assets-receipt.json',encode(answer))
    sync(RUN)
    print(json.dumps(dict(answer,receipt=receipt),sort_keys=True),flush=True)


def baseline():
    need(pin(read(ORIGINAL,65536))['SHA256'] == ORIGINAL_SHA, 'actual_previous_production_carrier')
    api = runpy.run_path(str(ORIGINAL))
    api['fresh'].__globals__['LABEL'] = LABEL
    return api


def worker(api):
    number = os.environ.get('SLURM_STEP_ID','')
    step = PARENT + '.' + number
    need(os.environ['SLURM_JOB_ID'] == PARENT and socket.gethostname().split('.')[0] == NODE
         and not os.environ.get('SLURM_STEP_GPUS') and number.isdecimal()
         and any({'job_'+PARENT, 'step_'+number} <= set(line.split(':',2)[-1].split('/'))
                 for line in Path('/proc/self/cgroup').read_text().splitlines()), 'actual_new_zero_GRES_cgroup')
    rows = [row for line in api['query'](['scontrol','show','step',step,'--oneliner'])
            if (row:=api['fields'](line)).get('StepId') == step]
    need(len(rows) == 1 and rows[0]['NodeList'] == NODE and rows[0]['State'] == 'RUNNING'
         and rows[0].get('Name',rows[0].get('JobName')) == LABEL
         and api['tres'](rows[0].get('TRES',rows[0].get('AllocTRES',''))), 'actual_controller_zero_GRES')
    shell = ('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh >&2; unset PYTHONPYCACHEPREFIX; '
             'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 '
             'PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '
             'PYTHONPATH=/training-runtime/site:/pilot/framework/generation/src:/pilot/framework; exec ' + shlex.join(
              ['python3','-B','-u','/audit-source/native_entry.py','--assets','/audit-assets',
               '--manifest-sha256',INPUT_SHA,'--step',step]))
    executable = '/public/software/apps/apptainer/1.3.4/bin/apptainer'
    argv = [executable, 'exec', '--no-home', '--pwd', '/']
    mounts = [(SRC,'/audit-source'),(ASSETS,'/audit-assets')]
    mounts.extend((path, dest) for path,dest in api['mapped']() if dest in ('/pilot/framework','/training-runtime','/opt/hyhal'))
    for path,dest in mounts:
        argv += ['--bind',f'{path}:{dest}:ro']
    argv += [str(ROOT/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
    os.chdir('/')
    os.execv(executable,argv)


def cpu(data):
    need(data['request'] == CPU, 'exact_new_CPU_scope')
    installed = installation(data['installed_SHA256'])
    asset_receipt = read(RUN/'assets-receipt.json',8192)
    need(pin(asset_receipt)['SHA256'] == data['asset_receipt_SHA256'], 'exact_asset_install')
    asset = json.loads(asset_receipt)
    need(asset['passed'] and asset['request'] == TRANSFER and asset['verified_objects'] == 79
         and asset['verified_manifest_bytes'] == EXPECTED_BYTES and asset['input_SHA256'] == INPUT_SHA
         and verify_tree(ASSETS,input_spec(read(SRC/'INPUT_FILES.json',65536))), 'exact_readback_before_native_run')
    api = baseline()
    command = ['srun','--jobid='+PARENT,'--exact','--immediate=5','--cpu-bind=none','-N1','-n1',
               '-c4','--mem=49152M','--gres=none','-w',NODE,'-t','00:03:00','--job-name='+LABEL,
               MANAGED,'-I','-S','-B','-u',str(RUN/'carrier.py'),'worker']
    with api['lock']():
        api['fresh']()
        begin = time.time()
        intent = {'request':CPU,'publication':installed['publication'],'argv':pin(encode(command)),
                  'epoch':begin,'deadline':begin+210,'asset_receipt_SHA256':data['asset_receipt_SHA256']}
        write(RUN/'cpu-intent.json',encode(intent));sync(RUN)
    result = {'request': CPU, 'passed': False, 'actual_GPU': 0, 'replay_allowed': False}
    try:
        p = subprocess.run(command,capture_output=True,timeout=185)
        need(len(p.stdout)<=32768 and len(p.stderr)<=65536,'bounded_CPU_result')
        result.update(rc=p.returncode,stdout=write(RUN/'cpu.stdout.private',p.stdout),
                      stderr=write(RUN/'cpu.stderr.private',p.stderr))
        rows = api['accounting']()
        until = time.monotonic()+10
        while (len(rows)!=1 or rows[0][2] in ('RUNNING','PENDING','COMPLETING')) and time.monotonic()<until:
            time.sleep(.3)
            rows=api['accounting']()
        result['accounting']=rows
        visible=[line[len(PREFIX):] for line in p.stdout.splitlines() if line.startswith(PREFIX)]
        if len(visible)==1:
            native = json.loads(visible[0]); result['native']=native
        else:
            native = {}
        result['passed']=(p.returncode==0 and len(rows)==1 and api['row_valid'](rows[0],intent)
                         and int(rows[0][4])<=180 and native.get('numeric_step')==rows[0][0]
                         and native.get('passed') is True and native['qualified_input_pairs']==19
                         and native['actual_GPU']==native['actual_predictions']==native['model_weights_opened']==0)
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
        result['error_type']=type(error).__name__
    receipt=write(RUN/'cpu-receipt.json',encode(result));sync(RUN)
    return dict(result,receipt=receipt)


def main():
    need(os.getuid()==154459, 'existing_project_UID')
    os.umask(0o077)
    if sys.argv[1:]==['worker']:
        worker(baseline())
    elif len(sys.argv)==7 and sys.argv[1]=='transfer':
        transfer(sys.argv[2:])
    else:
        data=json.load(sys.stdin)
        try:
            answer = {'stage':stage,'cpu':cpu}[data['mode']](data)
        except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as error:
            answer = {'request':data['request'],'passed':False,'error_type':type(error).__name__,
                      'reason_SHA256':pin(str(error).encode())['SHA256']}
        print(json.dumps(answer,sort_keys=True),flush=True)


if __name__=='__main__':
    main()
