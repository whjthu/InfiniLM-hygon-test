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
LABEL = 'training-gemma-usgs-r39-two-updates-native-v1'
ROOT = R / 'results' / LABEL
ASSET = R / 'results/training-gemma-usgs72-fresh-initial-domain25-v1'
CORE = R / 'incoming/training-gemma-usgs72-fresh-initial-domain25-v1-source'
CORE_PUBLICATION = '16081fc061703371d8e19012ded9e073a87f98c3'
CORE_MF_SHA = 'd3e8a6432e89db845e527a0d32b9ff39139416d2fb08885babcb5eba67156eb1'
STAGE_PIN = dict(bytes=151938, SHA256='f10d27753331d6ea085084af202a571a09880d69e86164e4c914cbcfbfe59fa0')
FRAMEWORK_SHA = '40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329'
RUNTIME_SHA = '2bf81432130fa7c69920753f1666ce5cafedaecefe941327a268ad86d6412b10'
TRAINING_REQUEST = '6d14e9df9832432b97f1f46f032b49f3'
REQUEST = TRAINING_REQUEST
DISPATCH_REQUEST = '1e7acbb2e9124766a63ba5e6a79df305'
PARENT, NODE, PORT = '1103606', 'f13r1n12', 29932
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
