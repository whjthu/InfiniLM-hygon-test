"""Input-SHA/split bound target lookup for a future explicit SF3D domain hook.

Returns the three original Python target fields, never fabricated features.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

D=Path(__file__).resolve().parent
G=D.parents[2]
FROZEN=G/'code/bw1000/training-sf3d-public-pbr-atlas2-v1'
BINDING=G/'local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original/complete-data-binding-private.json'
BINDING_PIN={'bytes':2101,'SHA256':'209bd0d1eec590589b39bc3481cecd4a08fb0a901afb5caf85932f09af5e5185'}


def checked(path,expected):
    raw=path.read_bytes()
    if {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}!=expected:raise ValueError('Actual source identity differs: '+path.name)
    return raw


def targets_for_image(input_path,*,recipe,purpose):
    if recipe!='sf3d-public-pbr-UV-atlas-v1':raise ValueError('Explicit independent domain recipe required')
    if purpose not in ('train','eval-dev'):raise ValueError('Explicit train or eval-dev purpose required')
    binding=json.loads(checked(BINDING,BINDING_PIN))
    selected=Path(input_path).resolve(strict=True)
    matches=[(split,row) for split,row in binding['objects'].items() if (G/row['input']).resolve(strict=True)==selected]
    if len(matches)!=1:raise ValueError('Input must bind to exactly one frozen object/split')
    split,row=matches[0]
    if split!=('train' if purpose=='train' else 'dev'):raise ValueError('Heldout object cannot enter training')
    # Split rejection happens before image/label body reads for the rejected row.
    checked(selected,row['input_pin'])
    manifest=json.loads(checked(FROZEN/'SOURCE_FILES.json',{'bytes':986,'SHA256':'9918612f5815bd99fad10321f20735a20a36cf9888adaa2979db04ca196a3b0f'}))
    for name in ('read_targets.py','pbr.py'):checked(FROZEN/name,manifest['files'][name])
    spec=importlib.util.spec_from_file_location('published_target_bundle_loader',FROZEN/'read_targets.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    fields=loader.load_targets(G/row['target'],row['target_pin'],recipe)
    # This is the original foreground_material_targets interface used at1127..29,
    # not a tensor cache. Explicit scalar .item() avoids implicit ndarray casts.
    target={'albedo':fields['albedo'][0].copy(),'roughness':fields['roughness'].item(),'metallic':fields['metallic'].item()}
    identity={'recipe':recipe,'purpose':purpose,'object_group':row['group'],'split':split,
        'input':row['input_pin'],'target_bundle':row['target_pin'],'binding':BINDING_PIN}
    identity['label_input_binding_SHA256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return target,identity
