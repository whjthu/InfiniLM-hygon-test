"""Only exact published source, processor metadata and scoped data are read."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL='training-treasury-timeseries-native-CPU-package-v1'
RUN=ROOT/'results'/LABEL
FRAMEWORK=ROOT/'results/training-qwen-a68952c-swift-initial-CPU-v1/framework'
MODEL=ROOT/'results/training-qwen-original-public-model-v1/model'


def require(ok,why):
    if not ok:raise ValueError(why)


def read(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=262144,"unsafe metadata")
    return json.loads(path.read_bytes())


def pin(path,expected):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size==expected['bytes'] and before.st_size<=16*1024**2,
            "source/processor metadata type or size differs")
    raw=path.read_bytes();after=path.lstat()
    require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))
            and hashlib.sha256(raw).hexdigest()==expected['SHA256'],"source/processor metadata SHA differs")
    return raw


def relative(name):
    value=PurePosixPath(name)
    require(not value.is_absolute() and value.as_posix()==name and '..' not in value.parts,"unsafe manifest path")
    return value


def exact_data(source,data):
    sys.path.insert(0,str(source/'native'))
    from check_native import bind
    _,manifest,rows,media=bind(data)
    expected={'manifest.json','train/labels.jsonl',*[s+'/inputs.jsonl' for s in ('train','dev','blind')],*media}
    actual=set()
    for i,path in enumerate(data.rglob('*')):
        require(i<80 and not path.is_symlink(),"unexpected data traversal or symlink")
        if path.is_file():actual.add(path.relative_to(data).as_posix())
    require(actual==expected and len(actual)==53,"native data-view file set differs")
    total=sum((data/name).stat().st_size for name in expected)
    require(total==3489522,"native data-view bytes differ")
    return dict(files=53,bytes=total,input_rows={s:len(v) for s,v in rows.items()},media_SHA_verified=len(media),
                dev_blind_labels_present=False,dev_blind_labels_content_opened=False,
                manifest_SHA256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())


def runtime(source):
    info=read(source/'RUNTIME_PINS.json')
    for name,expected in info['processor'].items():
        require(len(relative(name).parts)==1 and name.endswith(('.json','.txt')),"nonmetadata model member")
        pin(MODEL/name,expected)
    require(len(info['processor'])==9 and len(info['framework'])==393 and len(info['vision'])==7,
            "provider/source pin denominator differs")
    for name,expected in info['framework'].items():pin(FRAMEWORK/relative(name),expected)
    vision=ROOT/'results/training-qwen-vision014-CPU-v2/vision'
    for name,expected in info['vision'].items():pin(vision/relative(name),expected)
    pin(ROOT/'results/training-vlm-swift-offline-runtime-v1/site/swift/template/templates/qwen.py',info['template'])
    pin(ROOT/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site/transformers/models/qwen3_vl/modeling_qwen3_vl.py',info['qwen_model'])
    return info


def original_cpu330_accounting(accounting):
    return accounting('1087656','training-qwen-treasury-explicit-dev-CPU-v1')


def cpu330(source,accounting):
    info=read(source/'RUNTIME_PINS.json')['CPU330']
    base=ROOT/'results/training-qwen-treasury-explicit-dev-CPU-v1'
    paths={'intent':'intent.json','outcome':'outcome.json','source_manifest':'source/SOURCE_FILES.json','log':'step.log'}
    for key,name in paths.items():
        p=base/name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size<=4*1024**2
                and hashlib.sha256(p.read_bytes()).hexdigest()==info[key],"original CPU330 artifact differs")
    outcome=read(base/'outcome.json');row=original_cpu330_accounting(accounting)
    require(outcome['passed'] is True and outcome['accounting']['row']==row
            and row[:4]==['1087656.330','training-qwen-treasury-explicit-dev-CPU-v1','COMPLETED','0:0']
            and row[5]=='a01r4n14' and outcome['metrics']['actual_GPU_cards']==0,"original CPU330 no longer qualified")
    original=read(base/'source/SOURCE_FILES.json')['files']
    for name,expected in original.items():
        require(len(relative(name).parts)==1,'unsafe original CPU330 source member')
        pin(base/'source'/name,expected)
    return dict(handle=row[0],outcome_SHA256=info['outcome'],accounting=row)
