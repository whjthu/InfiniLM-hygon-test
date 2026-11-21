"""Real two-train/one-dev target-only joins; no cached feature construction."""
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath

D=Path(__file__).resolve().parent
G=D.parents[2]
BINDING=G/'local-only/training-sf3d-public-corset-train-atlas-v1-original/three-object-binding-private.json'
PIN={'bytes':1902,'SHA256':'54ecd851d91d48359be528d93632824f1ad7d553c257ef783df95a38991ba8e7'}
FROZEN=G/'code/bw1000/training-sf3d-public-pbr-atlas2-v1'
RECIPE='sf3d-public-pbr-UV-atlas-v1'


def checked(path,expected):
    raw=path.read_bytes()
    if {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}!=expected:raise ValueError('Frozen file identity differs: '+path.name)
    return raw


def local_path(relative):
    p=PurePosixPath(relative.replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or p.parts[0]!='local-only':raise ValueError('Expected exact private relative public-data path')
    result=(G/Path(*p.parts)).resolve(strict=True)
    if G.resolve() not in result.parents:raise ValueError('Public-data binding leaves project root')
    return result


def binding():
    value=json.loads(checked(BINDING,PIN))
    if value['schema']!='SF3D_PUBLIC_ATLAS_THREE_OBJECTS_V1' or value['prior_binding_SHA256']!='209bd0d1eec590589b39bc3481cecd4a08fb0a901afb5caf85932f09af5e5185':raise ValueError('Original binding chain differs')
    if [r['group'] for r in value['train']]!=['WaterBottle','Corset'] or [r['group'] for r in value['development']]!=['BoomBox'] or value['blind'] or value['native_features_generated'] is not False:raise ValueError('Frozen group split/feature status differs')
    return value


def targets_for_group(group,*,purpose,recipe=RECIPE):
    if recipe!=RECIPE or purpose not in ('train','eval-dev'):raise ValueError('Explicit independent recipe and split purpose required')
    b=binding();matches=[(split,row) for split in ('train','development') for row in b[split] if row['group']==group]
    if len(matches)!=1:raise ValueError('Unique frozen object group required')
    split,row=matches[0]
    if split!=('train' if purpose=='train' else 'development'):raise ValueError('Development group cannot enter training')
    checked(local_path(row['input']),row['input_pin'])
    m=json.loads(checked(FROZEN/'SOURCE_FILES.json',{'bytes':986,'SHA256':'9918612f5815bd99fad10321f20735a20a36cf9888adaa2979db04ca196a3b0f'}))
    for name in ('read_targets.py','pbr.py'):checked(FROZEN/name,m['files'][name])
    spec=importlib.util.spec_from_file_location('frozen_atlas_target_loader',FROZEN/'read_targets.py');loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    fields=loader.load_targets(local_path(row['target']),row['target_pin'],RECIPE)
    target={'albedo':fields['albedo'][0].copy(),'roughness':fields['roughness'].item(),'metallic':fields['metallic'].item()}
    return target,{'group':group,'split':split,'input':row['input_pin'],'target':row['target_pin'],'UV_slots':256}
