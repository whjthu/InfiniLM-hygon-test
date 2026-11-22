"""Fixed Gemma continuous-only scope; no fresh initial snapshot is transported."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

D = Path(__file__).resolve().parent
R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'training-gemma-usgs72-question-only-resume20-v2'
ROOT = R / 'results' / LABEL
ASSET = R / 'results/training-gemma-usgs72-question-only-control25-v1'
CORE = D
CORE_PUBLICATION = '2c470a93a42d8d9723f63c95e2a0644141c4ed7f'
CORE_MF_SHA = 'd94f952d1303961036f14f2c054ad5100569faf8297e87b0f1d766e16a6ae473'
STAGE_PIN = dict(bytes=150566, SHA256='02c31f7cc5b5f5c6dcdc24e173fc2ac3ba18c2acf44c9c1a597827f7d3afff65')
FRAMEWORK_SHA = '9b7b5be546b9f6b4c259a949e1ef2b4948397032375c2b1556f8ddbc972a7642'
RUNTIME_SHA = '2bf81432130fa7c69920753f1666ce5cafedaecefe941327a268ad86d6412b10'
TRAINING_REQUEST = '2ea01ac64fa07c326d825f0e073b254b'
REQUEST = TRAINING_REQUEST
DISPATCH_REQUEST = 'f0619ccf06774f798e345e9982965660'
CONTINUOUS_ROOT = R / 'results/training-gemma-usgs72-question-only-control25-v1-native'
CONTINUOUS_GATE_PIN = dict(bytes=8493,SHA256='f9956d84c779559d8b8b68de3ef24942fab35605832f99b209ac56802641a129')
CONTINUOUS_OUTCOME_SHA = 'f2d166439f7b38e3820da24cb8c91af7aeea8ca59fcfe11dc69007237be8e5e9'
CONTINUOUS_NATIVE_SHA = '4db77eea4cce17bf30868bb2d6f7ddc0aff483f7549ef8f17276f727b1630e9c'
PARENT, NODE, PORT = '1103606', 'f13r1n12', 29965
WHOLE, INSIDE, GPU, CLEANUP = 2400, 1980, 2100, 2080
COLUMNS = ['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES']


def need(ok, why):
    if not ok:
        raise ValueError(why)


class Clock:
    def __init__(self, epoch):
        now = time.time()
        need(type(epoch) in (int,float) and math.isfinite(epoch) and 0 < epoch <= now < epoch + WHOLE,
             'invalid original epoch')
        self.epoch = epoch
        self.end = time.monotonic() + epoch + WHOLE - now
    def left(self, offset=WHOLE, reserve=0):
        value = min(self.epoch + offset - time.time(), self.end - time.monotonic()) - reserve
        need(value > 0, 'original epoch exhausted')
        return value


def pin(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def stable_json(path, expected=None, cap=2*1024**2):
    path = Path(path)
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
    keys = lambda s: [getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]
    with path.open('rb') as f:
        opened = os.fstat(f.fileno()); raw = f.read(cap+1); after = os.fstat(f.fileno())
    need(keys(before)==keys(opened)==keys(after)==keys(path.lstat()) and len(raw)==before.st_size,
         'metadata identity changed')
    actual = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
    need(expected is None or actual == expected, 'metadata exact pin differs')
    return json.loads(raw), actual


def save(name, value, replace=False):
    target = ROOT / name
    if replace:
        target = ROOT / (name + '.tmp')
    raw = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(len(raw)<=2*1024**2,'bounded saved record')
    with target.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    if replace:
        target.replace(ROOT/name)
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def verify(kind):
    manifest = json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name, expected in manifest['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe package member')
        need(pin(D/name)==expected, 'operative package member differs')
    claim = json.loads((D/'CLAIM.json').read_bytes())
    need(claim['request_id']==DISPATCH_REQUEST and claim['training_request_id']==TRAINING_REQUEST and claim['root_GO'] is True and claim[kind+'_GO'] is True,
         'operative authorization false or wrong request')
    need(claim['Model_GO'] is False and claim['formal17_GO'] is False, 'scope qualification drift')
    if kind=='GPU':
        need(claim['training_GO'] is True, 'training authorization false')
    return pin(D/'SOURCE_FILES.json')


def parse_accounting(text):
    rows = [line.split('|') for line in text.splitlines() if line.strip()]
    need(all(len(row)==len(COLUMNS) for row in rows), 'malformed original accounting row')
    return rows


def parse_step_membership(text):
    rows=[line.strip().split('|') for line in text.splitlines() if line.strip()]
    need(len(rows)<=4096 and all(len(row)==2 for row in rows),'malformed squeue step columns')
    result=[]
    for handle,hosts in rows:
        handle,hosts=handle.strip(),hosts.strip()
        if handle in (PARENT+'.batch',PARENT+'.extern'):
            continue
        need(re.fullmatch(PARENT+r'\.\d+',handle) and hosts not in ('','(null)'),
             'unclassified active step membership')
        result.append((handle,hosts))
    return result


def allocated(value):
    entries = value.split(',')
    need(all('=' in v for v in entries), 'unclassified TRES')
    data = dict(v.split('=',1) for v in entries)
    need(len(data)==len(entries), 'duplicate TRES')
    generic=data.get('gres/dcu'); typed=[v for k,v in data.items() if k.startswith('gres/dcu:')]
    need(all(v.isdecimal() for v in typed+([generic] if generic is not None else [])), 'noninteger DCU TRES')
    if generic is not None:
        need(not typed or sum(map(int,typed))==int(generic), 'DCU TRES disagree')
        return int(generic)
    if typed:return sum(map(int,typed))
    return 0 if 'cpu' in data and not any(k.startswith('gres/') for k in data) else None


def step_argv(phase, payload, seconds):
    nominal = {'pre':60,'gpu':GPU,'post':90}[phase]
    need(type(seconds) in (int,float) and math.isfinite(seconds) and 60<=seconds<=nominal,
         'no bounded Slurm minute remains')
    minutes = int(seconds//60) # actual Slurm allocation cannot exceed remaining original epoch
    return ['srun','--jobid='+PARENT,'--exact','--immediate=5','--cpu-bind=none','-N1','-n1',
        '-w',NODE,'-c'+('64' if phase=='gpu' else '2'),
        '--mem='+('224000M' if phase=='gpu' else '4096M'),
        '--gres='+('dcu:8' if phase=='gpu' else 'none'),'-t',str(minutes),
        '--job-name='+LABEL+'-'+phase,*payload]


def numbered(row):
    return row is not None and re.fullmatch(PARENT+r'\.\d+',row[0]) is not None


def validate_prior_gate(gate):
    need(gate.get('request_id')==TRAINING_REQUEST and gate.get('passed') is True
         and gate.get('independent_passed') is True and gate.get('accounting_C0') is True
         and gate.get('owned_cleanup_passed') is True and gate.get('actual_updates')==25
         and gate.get('arm')=='continuous' and gate.get('ranks_completed')==8
         and gate.get('microbatches_per_rank')==100,'original continuous25 independent gate incomplete')
    names={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json',
           'optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))}
    need(set(gate.get('checkpoint20_files',{}))==names,'original checkpoint20 required14 pins absent')
    for p in gate['checkpoint20_files'].values():
        need(type(p['bytes']) is int and 0<p['bytes']<=512*1024**2
             and isinstance(p['SHA256'],str) and re.fullmatch('[0-9a-f]{64}',p['SHA256']),
             'original checkpoint20 pin malformed')
    return gate


def prior_continuous():
    gate,_=stable_json(CONTINUOUS_ROOT/'independent-terminal.json',CONTINUOUS_GATE_PIN)
    validate_prior_gate(gate)
    _,outcome_pin=stable_json(CONTINUOUS_ROOT/'outcome.json')
    _,native_pin=stable_json(CONTINUOUS_ROOT/'native-outcome.json')
    need(outcome_pin['SHA256']==CONTINUOUS_OUTCOME_SHA and native_pin['SHA256']==CONTINUOUS_NATIVE_SHA,
         'original1206 native/outcome identity changed')
    return gate
