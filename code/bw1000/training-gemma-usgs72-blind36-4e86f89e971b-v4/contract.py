"""Fixed Gemma continuous-only scope; no fresh initial snapshot is transported."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from decimal import Decimal
from datetime import datetime, timezone

D = Path(__file__).resolve().parent
R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'training-gemma-usgs72-blind36-4e86f89e971b-v4'
ROOT = R / 'results' / LABEL
ASSET = ROOT / 'inference'
DOMAIN = R / 'results/training-gemma-usgs72-fresh-initial-domain25-v1'
SOURCE_ASSETS = R / 'results/training-vlm-original-assets-v1'
CORE = D / 'core'
CORE_PUBLICATION = None
CORE_MF_SHA = None
STAGE_PIN = FRAMEWORK_SHA = RUNTIME_SHA = None
TRAINING_REQUEST = '4e86f89e971b4d62ae55ed80c9f264b5'
REQUEST = TRAINING_REQUEST
DISPATCH_REQUEST = '2aeb75d64acd429a9e13e5b89ebfe679'
STAGE_LABEL = 'training-gemma-usgs72-question-only-control25-v1-stage'
PARENT, NODE, PORT = '1103606', 'f13r1n12', 29984
WHOLE, INSIDE, GPU, CLEANUP = 900, 660, 720, 735
COLUMNS = ['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList','AllocTRES','Start','End']


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
    manifest=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,expected in manifest['files'].items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts and '\\' not in name,'unsafe delivered member')
        need(pin(D/name)==expected,'operative outer member differs')
    claim=json.loads((D/'CLAIM.json').read_bytes())
    need(claim['request_id']==DISPATCH_REQUEST and claim['core_request']==REQUEST and
         claim['root_GO'] is claim[kind+'_GO'] is False and claim['training_GO'] is claim['Model_GO'] is claim['formal17_GO'] is False,
         'operative outer flags false/wrong scope')
    nested=json.loads((CORE/'CLAIM.json').read_bytes())
    need(all(nested[k] is claim[k] for k in ('root_GO','GPU_GO','READ_GO','SCORE_GO','training_GO','Model_GO','formal17_GO')),
         'nested/outer operative flag mismatch')
    mf=json.loads((CORE/'SOURCE_FILES.json').read_bytes())
    for name,expected in mf['files'].items():
        need(Path(name).name==name and pin(CORE/name)==expected,'operative nested source differs')
    need(nested['request']==REQUEST and nested['read_request']=='f52370e71c7641a6a1304d3f631fb49'
         and nested['score_request']=='98ad7df5613f407faad9f333a741bbae','fixed original core requests differ')
    return pin(D/'SOURCE_FILES.json')


def core_source():return pin(CORE/'SOURCE_FILES.json')


def accounting_argv():
    return ['env','TZ=UTC','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S','sacct','-n','-P','-j',PARENT,
            '-o',','.join(COLUMNS).replace('JobName','JobName%120')]


def exact_label(records,phase):
    label=LABEL+'-'+phase
    found=[row for row in records if row[1]==label]
    need(len(found)<=1,'duplicate exact phase label')
    if found:
        need(numbered(found[0]) and found[0][5]==NODE,'label belongs to wrong parent/numbered node')
        return found[0]
    return None


def accounting_window(row,step_intent):
    try:
        if (len(row)!=len(COLUMNS) or not row[4].isdigit() or
                not all(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d',value) for value in row[7:9])):
            return False
        start,end=[datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() for value in row[7:9]]
        epoch,submit=step_intent['epoch'],step_intent['submitted_at']
        seconds,limit=int(row[4]),step_intent['slurm_seconds']
        return (epoch<=submit<=epoch+WHOLE and submit-1<=start<=end<=epoch+WHOLE+1
                and 0<=seconds<=limit and abs((end-start)-seconds)<=1)
    except (KeyError,TypeError,ValueError):
        return False


def parse_accounting(text):
    need(len(text)<=2<<20,"bounded accounting response required")
    rows = [line.split('|') for line in text.splitlines() if line.strip()]
    need(len(rows)<=8192 and all(len(row)==len(COLUMNS) for row in rows), 'malformed original accounting row')
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
    nominal = {'pre':60,'gpu':GPU,'post':60,'read':60}[phase]
    need(type(seconds) in (int,float) and math.isfinite(seconds) and 60<=seconds<=nominal,
         'no bounded Slurm minute remains')
    minutes = int(seconds//60) # actual Slurm allocation cannot exceed remaining original epoch
    return ['srun','--jobid='+PARENT,'--exact','--immediate=5','--cpu-bind=none','-N1','-n1',
        '-w',NODE,'-c'+('8' if phase=='gpu' else '2'),
        '--mem='+('64000M' if phase=='gpu' else '4096M'),
        '--gres='+('dcu:1' if phase=='gpu' else 'none'),'-t',str(minutes),
        '--job-name='+LABEL+'-'+phase,*payload]


def numbered(row):
    return row is not None and re.fullmatch(PARENT+r'\.\d+',row[0]) is not None

def resources_match(phase,tres):
    fields=dict(item.split('=',1) for item in tres.split(','))
    memory=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGTP])',fields.get('mem',''))
    if memory is None:return False
    mib=Decimal(memory[1])*{'K':Decimal(1)/1024,'M':Decimal(1),'G':Decimal(1024),
                          'T':Decimal(1024)**2,'P':Decimal(1024)**3}[memory[2]]
    return (allocated(tres)==(1 if phase=='gpu' else 0) and fields.get('cpu')==('8' if phase=='gpu' else '2')
            and fields.get('node')=='1' and mib==(64000 if phase=='gpu' else 4096))
