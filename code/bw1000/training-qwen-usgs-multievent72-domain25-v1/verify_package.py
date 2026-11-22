"""Pure source/operative-claim gate; no device, data, model or network access."""
import ast
import hashlib
import json
from pathlib import Path

REQUEST='c933ba7359f84e8b9b80ac40d2d30964'


def verify(directory,adopted=False):
    directory=Path(directory)
    mf=json.loads((directory/'SOURCE_FILES.json').read_bytes())
    for name,pin in mf['files'].items():
        p=directory/name;raw=p.read_bytes()
        if p.is_symlink() or p.parent!=directory or len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['SHA256']:
            raise ValueError('exact frozen package member differs: '+name)
        if name.endswith('.py'):ast.parse(raw,feature_version=(3,10))
    c=json.loads((directory/'CLAIM.json').read_bytes())
    if (c['request']!=REQUEST or c['label']!=directory.name or c['optimizer_updates']!=25 or
        c['updates_by_arm']!={'continuous':25} or c['train_source_rows']!=72 or
        c['training_exposure_rows']!=800 or c['unique_train_media']!=12 or
        c['whole_seconds']!=2400 or c['gpu_seconds']!=1800 or c['native_seconds']!=1700 or
        c['cpu_native_seconds']!=300 or c['dev_metric_rows']!=36 or c['dev_metric_predictions']!=72 or
        any(c[k] is not adopted for k in ('root_GO','CPU_GO','GPU_GO','training_GO','READ_GO','metric_inference_GO','SCORE_GO')) or
        any(c[k] is not False for k in ('Model_GO','formal_training_GO','effect_GO','performance_GO','training_READY'))):
        raise ValueError('new scope or operative adoption flags differ')
    repo=directory.parents[2]
    for item in c['public_prerequisites']:
        raw=(repo/item['path']).read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['SHA256']:
            raise ValueError('published prerequisite differs')
    return dict(passed=True,members=len(mf['files']),adopted=adopted,
                actual_GPU=0,actual_new_native_processor=0,actual_optimizer_updates=0)
