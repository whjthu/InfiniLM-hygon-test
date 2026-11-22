"""Fixed existing-data Gemma control scope. No model/network imports."""
import hashlib
import json
import math
from pathlib import Path,PurePosixPath

D=Path(__file__).resolve().parent
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
ROOT=R/'results/training-gemma-usgs72-question-only-control25-v1'
SOURCE_ASSETS=R/'results/training-vlm-original-assets-v1'
DOMAIN=R/'results/training-gemma-usgs72-fresh-initial-domain25-v1'
TRAINING_REQUEST='2ea01ac64fa07c326d825f0e073b254b'
STAGE_REQUEST='f21f69032dd0aa610d069f7cb8da5210'
PARENT,NODE,PORT='1103606','f13r1n12',29961
INPUT_SHA='21637fd30cab21ef005c64d6eddb09657e6933fd7817c75cba75622a1aaa79fa'
INPUT_BYTES=43014
GATE_PIN=dict(bytes=7528,SHA256='ba9e7917eb04b298de2104aa5fce80f38b19aa39486d18a2f1f681f446c07a09')
FRESH_NAMED_SHA='e6b624e3342df4d84751be5b65c2722137ff17109f2dfb18db8c5fd4c916b98c'
FRAMEWORK_SHA='40ed7488227fffc4da0a34353fc711c8720c0ebc407040fc61e748b2a7d24329'
RUNTIME_SHA='2bf81432130fa7c69920753f1666ce5cafedaecefe941327a268ad86d6412b10'
SOURCE_INPUT_MANIFEST_SHA='85b1cc6b6ec40d76d26f7950c7d2e6d7927af654e32f29e3e86149675787afcf'
VIEW_SHA='a2d2411778774856889797c3f1851b6f5df7fe67fd6a81ff113ab03d3da3d144'
EXPOSURE_LEDGER_SHA='9e2ab684c5ae9079d6e593388da64ab729d56a7dc4c2d94b1385da4d48a2eaa6'


def need(ok,reason):
    if not ok:raise ValueError(reason)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def identity(value):return [getattr(value,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')]


def relative(name):
    p=PurePosixPath(name)
    need(p.as_posix()==name and not p.is_absolute() and '..' not in p.parts and '\\' not in name,'unsafe scoped member')
    return p


def clock(origin,deadline,now,maximum):
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (origin,deadline,now,maximum)) and
         0<origin<=now<deadline<=origin+maximum,'original clock invalid/exhausted')
    return deadline-now


def verify(kind):
    manifest=(D/'SOURCE_FILES.json').read_bytes()
    for name,expected in json.loads(manifest)['files'].items():
        relative(name);p=D/name;raw=p.read_bytes()
        need(not p.is_symlink() and dict(bytes=len(raw),SHA256=sha(raw))==expected,'source member differs: '+name)
    claim=json.loads((D/'CLAIM.json').read_bytes())
    need(claim['training_request']==TRAINING_REQUEST and claim['stage_request']==STAGE_REQUEST and
         claim['root_GO'] is claim[kind+'_GO'] is True,'operative scope not adopted')
    need(claim['Model_GO'] is claim['formal17_GO'] is claim['READY'] is False,'qualification drift')
    if kind=='GPU':need(claim['training_GO'] is True,'positive-update scope not adopted')
    return dict(bytes=len(manifest),SHA256=sha(manifest))
