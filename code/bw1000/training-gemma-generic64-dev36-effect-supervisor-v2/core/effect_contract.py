"""Gold-free fixed Gemma dev36 scope; all future observations require exact pins."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

D=Path(__file__).resolve().parent
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
DOMAIN=R/'results/training-gemma-usgs72-fresh-initial-domain25-v1'
DOMAIN_OP=R/'results/training-gemma-usgs72-continuous25-supervisor-v2'
GENERIC=R/'results/training-gemma-generic64-control25-v1'
GENERIC_OP=R/'results/training-gemma-generic64-control25-supervisor-v1'
ROOT=R/'results/training-gemma-generic64-dev36-effect-v2'
REQUEST='8d6304c0da154864aa036e8129d53ac5'
READ_REQUEST='40c0a366cd6a4eeaa11fb689be92e688'
SCORE_REQUEST='30205be083fb49ebad1e55596796d4ce'
ARMS=('base','domain25','generic25')
CAP=256
NAMED='e6b624e3342df4d84751be5b65c2722137ff17109f2dfb18db8c5fd4c916b98c'
DOMAIN_GATE=dict(bytes=7528,SHA256='ba9e7917eb04b298de2104aa5fce80f38b19aa39486d18a2f1f681f446c07a09')
DOMAIN_CP=dict(bytes=2977,SHA256='897b5aadef83c6c4cdab76d2a3d9a24c83e3eb47c6e97a48e7f71e8e24b3515c')
GENERIC_GATE=dict(bytes=8507,SHA256='261fc4ebbd519bb5126243440c3f65691d5b3a795331c320ba37ce5b75d94184')
GENERIC_CP=dict(bytes=2977,SHA256='0e484f8efe89b58db5c6251954b9bc5e25e5098a3b4429834954d7cc7cf83ae3')
FRAMEWORK_SHA='40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329'
RUNTIME_SHA='2bf81432130fa7c69920753f1666ce5cafedaecefe941327a268ad86d6412b10'

def need(ok,reason):
    if not ok:raise ValueError(reason)

def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())

def identity(s):return [getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]

def open_identity_matches(before,opened,windows):
    # Windows lstat/fstat report different ctime semantics. Preserve both clocks
    # within their own API provenance; Linux still requires all five across APIs.
    return before[:4]==opened[:4] and (windows or before==opened)

def read(path,expected=None,cap=2<<20,allow_empty=False):
    p=Path(path);before=p.lstat()
    need(stat.S_ISREG(before.st_mode) and (before.st_size>0 or allow_empty) and before.st_size<=cap,'bounded regular exact member required')
    with p.open('rb') as f:
        opened=os.fstat(f.fileno())
        need(open_identity_matches(identity(before),identity(opened),os.name=='nt'),'opened metadata lifetime differs')
        raw=f.read(cap+1)
        need(identity(os.fstat(f.fileno()))==identity(opened),'metadata changed during read')
    need(identity(p.lstat())==identity(before) and len(raw)==before.st_size and
         (expected is None or pin(raw)=={k:expected[k] for k in ('bytes','SHA256')}),'exact member pin/lifetime differs')
    return raw

def dump(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(raw)

def verify(kind):
    mf=read(D/'SOURCE_FILES.json')
    for name,expected in json.loads(mf)['files'].items():
        need(Path(name).name==name and '\\' not in name,'flat delivered source members required')
        read(D/name,expected)
    c=json.loads(read(D/'CLAIM.json'))
    need(c['request']==REQUEST and c['read_request']==READ_REQUEST and c['score_request']==SCORE_REQUEST and
         c['root_GO'] is c[kind+'_GO'] is True and c['Model_GO'] is c['formal17_GO'] is False,'operative scope not adopted')
    return pin(mf)

def clock(epoch,deadline,now):
    need(all(type(v) in (int,float) and math.isfinite(v) for v in (epoch,deadline,now)) and
         0<epoch<=now<deadline<=epoch+900,'one original finite900 epoch required')
    return deadline-now

def binding_scope(value,source):
    need(value['state']=='ACTUAL_THREE_ARM_GEMMA25_BINDING' and value['request']==REQUEST and
         value['source_manifest']==source and value['domain_terminal']==DOMAIN_GATE and value['generic_terminal']==GENERIC_GATE and
         value['initial_named_SHA256']==NAMED and value['initial']['path']==
         str(DOMAIN/'continuous/train/continuous-witness/rank-00/target-fresh-initial.pt') and
         value['checkpoints']['domain25']['root']==str(DOMAIN/'continuous/train/checkpoint-25') and
         value['checkpoints']['generic25']['root']==str(GENERIC/'continuous/train/checkpoint-25') and
         value['framework_root']==str(DOMAIN/'framework') and value['framework']['SHA256']==FRAMEWORK_SHA and
         value['runtime']['SHA256']==RUNTIME_SHA and value['model_root']==
         str(R/'results/training-gemma-original-model-v1/gemma-3-4b-it') and
         value['planned_each']==36 and value['planned_total']==108 and value['generation_cap']==CAP and
         value['dev_gold_read'] is value['blind_read'] is False,'exact actual three-arm source scope required')

def checkpoint(record,root,request):
    names={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json',
           'optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))}
    need(record['request_id']==request and record['step']==25 and record['required_members']==14 and
         set(record['files'])==names,'actual original checkpoint25 fourteen-member schema required')
    for name,value in record['files'].items():
        need(type(value['bytes']) is int and value['bytes']>0 and len(value['SHA256'])==64 and
             len(value['file_identity'])==5 and value['file_identity'][2]==value['bytes'],'checkpoint exact pin/identity required')
    return dict(root=str(root),files=record['files'])

def eos(ids,eos_ids):
    need(isinstance(ids,list) and 0<len(ids)<=CAP and all(type(x) is int and x>=0 for x in ids) and
         isinstance(eos_ids,list) and eos_ids and all(type(x) is int and x>=0 for x in eos_ids),'actual raw generation IDs/EOS required')
    cap=len(ids)>=CAP
    return dict(natural_EOS=not cap and ids[-1] in eos_ids and not any(x in eos_ids for x in ids[:-1]),
                cap_hit=cap,generated_tokens=len(ids),generated_ids_SHA256=pin(json.dumps(ids).encode())['SHA256'])
