"""Independent stored512 UV label comparison: scalar Pillow+Python glTF formula.

No generator/PBR helper import, HTTP, model, feature computation or new labels.
"""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

G=Path(__file__).resolve().parents[3]
OLD=G/'local-only/training-sf3d-public-pbr-atlas2-v1-original'
RUN=G/'local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original'


def fixed(path,size,sha):
    raw=path.read_bytes()
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('actual stored source identity differs')
    return raw


def main():
    result=json.loads(fixed(RUN/'result.json',3024,'0c8de07246cae77cfd8102ca843a7cc61ea30f915d6e115fdcdb94d168e348b3'))
    if result['passed'] is not True or result['completed_UV_slots']!=512:raise ValueError('original completion not qualified')
    binding=json.loads(fixed(RUN/'complete-data-binding-private.json',2101,'209bd0d1eec590589b39bc3481cecd4a08fb0a901afb5caf85932f09af5e5185'))
    specs=[('WaterBottle','train',2727,'0596f4e61dc781439d254fdfb5e3462daf1762c18715e3e3ac13001aa8f3f547',
        OLD/'WaterBottle/glTF/WaterBottle_baseColor.png',2165850,'62d0ad9b4e2c75e9bc7d67644a50d697563000ffa586d2e185ea38f5000e0a63',
        OLD/'WaterBottle/glTF/WaterBottle_occlusionRoughnessMetallic.png',3577832,'10cc809d9762e8c671f1ae73618dd5fdd07792d2c6ec9c4691cdb991617c9369'),
        ('BoomBox','dev',2702,'b1d2e0e56864c71a4b65fc560bfc1572b1eb32cfb9c8e86d06e6f9ace2803a43',
        OLD/'BoomBox/glTF/BoomBox_baseColor.png',3285844,'099816a7afc5f6690494313ac8039806fd6d5b84179126a808b2678aaab3563a',
        RUN/'BoomBox_occlusionRoughnessMetallic.png',4471451,'496704a4836ff364dc4441f41651edb25178498926c859933e33df42d6361412')]
    hashes=set();verified=0;error=0.;groups={}
    for name,split,gs,gh,bp,bs,bh,mp,ms,mh in specs:
        gltf=json.loads(fixed(OLD/name/'glTF'/f'{name}.gltf',gs,gh));pbr=gltf['materials'][0]['pbrMetallicRoughness']
        factor=pbr.get('baseColorFactor',[1.,1.,1.,1.]);rf=pbr.get('roughnessFactor',1.);mf=pbr.get('metallicFactor',1.)
        with Image.open(BytesIO(fixed(bp,bs,bh))) as bi,Image.open(BytesIO(fixed(mp,ms,mh))) as mi:
            base=bi.convert('RGBA');orm=mi.convert('RGB');base.putalpha(255)
            def rgba_pin(image):
                raw=image.convert('RGBA').tobytes();return hashlib.sha256(raw).hexdigest()
            a,b=rgba_pin(base),rgba_pin(orm)
            if a!=binding['four_texture_pixel_pins'][split+'-baseColor']['RGBA_pixels_SHA256'] or b!=binding['four_texture_pixel_pins'][split+'-ORM']['RGBA_pixels_SHA256']:raise ValueError('actual decoded pixel identity differs')
            hashes.update([a,b]);target=binding['objects'][split];q=target['target_pin']
            with np.load(BytesIO(fixed(G/target['target'],q['bytes'],q['SHA256'])),allow_pickle=False) as z:
                if z['UV'].shape!=(256,2) or z['albedo'].shape!=(256,3) or z['roughness'].shape!=(256,) or z['metallic'].shape!=(256,):raise ValueError('all256 stored slots required')
                for slot in range(256):
                    row,col=divmod(slot,16);u,v=(col+.5)/16,(row+.5)/16
                    if not np.array_equal(z['UV'][slot],[u,v]):raise ValueError('preselected UV location changed')
                    bx,by=int(u*base.width),int(v*base.height);mx,my=int(u*orm.width),int(v*orm.height)
                    rgb=base.getpixel((bx,by));packed=orm.getpixel((mx,my))
                    linear=[]
                    for i,c in enumerate(rgb[:3]):
                        x=c/255.;linear.append((x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4)*factor[i])
                    expected=linear+[packed[1]/255.*rf,packed[2]/255.*mf]
                    actual=list(z['albedo'][slot])+[z['roughness'][slot],z['metallic'][slot]]
                    if not np.isfinite(actual).all() or min(actual)<0 or max(actual)>1:raise ValueError('actual stored labels invalid')
                    error=max(error,max(abs(float(x)-y) for x,y in zip(actual,expected)));verified+=1
            groups[split]={'object_groups':1,'verified_UV_labels':256}
    if len(hashes)!=4 or verified!=512 or error>1e-6:raise ValueError('full label/dedup audit failed')
    r={'schema':'SF3D_PBR512_SAVED_LABEL_INDEPENDENT_MATH_V1','passed':True,'planned_UV_slots':512,'verified_UV_slots':verified,
        'missing_or_failed_slots':0,'independent_formula_max_abs_error':error,'frozen_tolerance':1e-6,
        'crossgroup_four_decoded_RGBA_unique':len(hashes),'groups':groups,'blind_groups':0,
        'generator_or_PBR_module_imported':False,'new_HTTP_or_features_or_model_or_training':0,'READY':False}
    with (RUN/'independent-label-audit.json').open('x',encoding='utf8') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(r,sort_keys=True))


if __name__=='__main__':main()
