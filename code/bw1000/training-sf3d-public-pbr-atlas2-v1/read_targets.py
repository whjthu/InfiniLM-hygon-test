"""Explicit opt-in target-only bridge to the frozen SF3D cache target schema."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np


def load_targets(path,expected,recipe):
    if recipe!='sf3d-public-pbr-UV-atlas-v1':raise ValueError('Separate domain recipe must explicitly opt in; original8 recipe cannot inherit these labels')
    raw=Path(path).read_bytes()
    if {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}!=expected:raise ValueError('Actual target bundle identity differs')
    with np.load(BytesIO(raw),allow_pickle=False) as z:
        fields={name:z['native_batch_'+name] for name in ('albedo','roughness','metallic')}
        if z['UV'].shape!=(256,2) or z['albedo'].shape!=(256,3) or z['roughness'].shape!=(256,) or z['metallic'].shape!=(256,):raise ValueError('All256 UV labels required')
    for name,a in fields.items():
        if a.dtype!=np.float32 or a.shape!=((1,3) if name=='albedo' else (1,)) or not np.isfinite(a).all() or (a<0).any() or (a>1).any():raise ValueError('Frozen native material target field contract differs')
    return fields


def main():
    g=Path(__file__).resolve().parents[3]
    root=g/'local-only/training-sf3d-public-pbr-atlas2-v1-original'
    expected={'bytes':2930,'SHA256':'309475d22858a0d2659519f27aba91a721045e069364f1b3c52674d46f623eea'}
    path=root/'WaterBottle/derived/material-targets-private.npz'
    fields=load_targets(path,expected,'sf3d-public-pbr-UV-atlas-v1')
    # Actual persisted UV slots, range and deterministic metadata; no self-score
    # is presented as a model result and no encoder feature is fabricated.
    with np.load(path,allow_pickle=False) as z:
        uv=z['UV'];slots={k:z[k] for k in ('albedo','roughness','metallic')}
        if not np.isfinite(uv).all() or len(np.unique(uv,axis=0))!=256 or not (uv>0).all() or not (uv<1).all():raise ValueError('Fixed UV grid missing/repeated')
        if any(not np.isfinite(v).all() or (v<0).any() or (v>1).any() for v in slots.values()):raise ValueError('Invalid real labels')
    rejected=0
    for bad_recipe,bad_pin in [('T11.material-head',expected),('sf3d-public-pbr-UV-atlas-v1',{'bytes':2930,'SHA256':'0'*64})]:
        try:load_targets(path,bad_pin,bad_recipe)
        except ValueError:rejected+=1
    if rejected!=2:raise ValueError('Required opt-in/identity boundary was not enforced')
    result={'schema':'ACTUAL_NEW_PBR_TARGET_NATIVE_FIELD_BRIDGE_V1','passed':True,'actual_objects':1,'actual_unique_UV_slots':256,
        'input_bundle':expected,'fields':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'finite_and_0to1':True} for k,a in fields.items()},
        'explicit_recipe_and_bad_identity_rejections':rejected,'allow_pickle':False,'triplane_or_image_features_created':False,
        'full_native_cache_or_Torch_import_qualified':False,'actual_model_updates':0,'READY':False}
    with (root/'target-only-bridge-result.json').open('x',encoding='utf8') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
