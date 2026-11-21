"""Two preselected official CC0 assets; only textures+metadata, no geometry/weights."""
import argparse
import ast
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tarfile
import time
import urllib.request
import urllib.error

D=Path(__file__).resolve().parent
G=D.parents[2]
BASE='https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/'
GROUPS=[('WaterBottle','train'),('BoomBox','dev')]
LIMIT=16*1024**2


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(b):return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())


def save(root,name,obj):
    raw=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    p=root/name
    with p.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(raw)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline',required=True,type=float);a=p.parse_args();root=Path(a.root)
    while not (root/'job-gate.json').exists():
        need(time.time()<a.deadline,'Job gate expired');time.sleep(.01)
    gate=json.loads((root/'job-gate.json').read_bytes());need(gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648,'Job gate differs')
    import numpy as np
    from PIL import Image
    from pbr import atlas_targets,uniform_uv_sample
    result={'schema':'SF3D_PUBLIC_PBR_ATLAS2_REAL_DATA_V1','passed':False,'planned_objects':2,'completed_objects':0,'failed_objects':0,
        'planned_PNGs':4,'completed_PNGs':0,'requests':[],'download_bytes':0,'objects':{},'planned_UV_slots':512,
        'actual_UV_slots':0,'actual_GPU_or_Torch_or_weight_or_training_calls':0,'READY':False,
        'original_T11_checkpoint_transfers_touched':False,'old8_RGBA_replay_or_formula_changed':False,
        'raw_commit':'UNKNOWN; retrieved official main snapshot frozen by actual body SHA, not claimed immutable commit'}
    opener=urllib.request.build_opener(NoRedirect())
    def fetch(asset,relative,cap):
        need(time.time()<a.deadline and len(result['requests'])<10,'original epoch/request budget')
        url=BASE+asset+'/'+relative;need(url not in [r['URL'] for r in result['requests']],'URL cannot be replayed')
        record={'URL':url,'attempted':True,'status':None,'read_bytes':0,'saved_bytes':0,'EOF':False}
        result['requests'].append(record);save(root,'progress.json',result)
        destination=root/asset/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        try:
            with opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=min(20,a.deadline-time.time())) as response:
                record.update(status=response.status,Content_Type=response.headers.get('Content-Type'),Content_Length=response.headers.get('Content-Length'),ETag=response.headers.get('ETag'))
                need(response.status==200 and response.headers.get('Content-Encoding','identity')=='identity','original200 unencoded body required')
                cl=record['Content_Length'];need(cl is None or cl.isdigit() and 0<int(cl)<=cap,'declared body exceeds cap before read')
                parts=[]
                with destination.open('xb') as f:
                    while True:
                        need(time.time()<a.deadline,'original download epoch exhausted')
                        b=response.read1(min(65536,cap+1-record['read_bytes']))
                        if not b:record['EOF']=True;break
                        record['read_bytes']+=len(b);result['download_bytes']+=len(b)
                        need(record['read_bytes']<=cap and result['download_bytes']<=LIMIT,'body/aggregate cap')
                        f.write(b);record['saved_bytes']+=len(b);parts.append(b)
                    f.flush();os.fsync(f.fileno())
                raw=b''.join(parts);need(len(raw)>0 and record['EOF'] and (cl is None or len(raw)==int(cl)),'natural EOF/length mismatch')
                record.update(pin(raw));return raw
        except urllib.error.HTTPError as e:
            record.update(status=e.code,error_type=type(e).__name__);e.close();raise
        except Exception as e:record['error_type']=type(e).__name__;raise
        finally:save(root,'progress.json',result)
    # Same original frozen backend, only this NumPy function; no fake Torch module.
    archive=G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
    need(pin(archive.read_bytes())=={'bytes':738802,'SHA256':'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'},'original framework archive differs')
    with tarfile.open(archive,'r:gz') as t:backend=t.extractfile('framework/generation/src/gencuda/backends/sf3d_training.py').read()
    need(pin(backend)=={'bytes':58458,'SHA256':'dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802'},'frozen SF3D source differs')
    function=next(n for n in ast.parse(backend).body if isinstance(n,ast.FunctionDef) and n.name=='foreground_material_targets')
    ns={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen-sf3d-foreground-target-function','exec'),ns)
    result['original_native_function']={'backend':pin(backend),'name':function.name,'AST_SHA256':hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest(),'whole_module_or_Torch_imported':False}
    seen_files=set();seen_pixels=set()
    for asset,split in GROUPS:
        record={'split':split,'passed':False,'group':asset,'native_input_or_model_qualified':False}
        result['objects'][asset]=record
        try:
            metadata=json.loads(fetch(asset,'metadata.json',65536));license_body=fetch(asset,'LICENSE.md',16384)
            need(metadata['version']==2 and any(x.get('license')=='CC0-1.0' and x.get('artist')=='Microsoft' and x.get('what')=='Everything' for x in metadata['legal']),'actual publisher rights differ')
            need(b'CC0-1.0' in license_body and b'CC-BY-4.0' in license_body,'model and metadocument licenses must both be retained')
            raw_gltf=fetch(asset,'glTF/'+asset+'.gltf',256*1024);gltf=json.loads(raw_gltf)
            need(gltf['asset']['version']=='2.0' and len(gltf['materials'])==1 and not gltf.get('extensionsRequired'),'single core material only')
            material=gltf['materials'][0];pbr=material['pbrMetallicRoughness']
            need(not material.get('extensions') and pbr.get('baseColorFactor',[1,1,1,1])==[1,1,1,1],'unmodified base RGB atlas scope requires unit factor')
            need(all('COLOR_0' not in p['attributes'] for m in gltf.get('meshes',[]) for p in m['primitives']),'vertex-color-modulated material needs new scope')
            texture={}
            for key,suffix in [('baseColorTexture','baseColor'),('metallicRoughnessTexture','occlusionRoughnessMetallic')]:
                binding=pbr[key];need(binding.get('texCoord',0)==0 and not binding.get('extensions'),'fixed UV0 with no transform')
                uri=gltf['images'][gltf['textures'][binding['index']]['source']]['uri']
                need(uri==asset+'_'+suffix+'.png' and re.fullmatch(r'[A-Za-z0-9_]+\.png',uri),'publisher texture binding differs')
                raw=fetch(asset,'glTF/'+uri,4*1024**2);need(pin(raw)['SHA256'] not in seen_files,'cross-object raw content collision');seen_files.add(pin(raw)['SHA256'])
                with Image.open(BytesIO(raw)) as image:
                    need(image.format=='PNG' and 0<image.width<=4096 and 0<image.height<=4096,'bounded actual PNG required');image.load()
                    texture[key]=np.asarray(image.convert('RGBA' if key=='baseColorTexture' else 'RGB')).copy()
                result['completed_PNGs']+=1
            rgba=texture['baseColorTexture'];rgba[...,3]=255 # glTF OPAQUE ignores stored alpha.
            pixelhash=hashlib.sha256(rgba.tobytes()).hexdigest();need(pixelhash not in seen_pixels,'cross-object input pixel collision');seen_pixels.add(pixelhash)
            targets,extrema,count=atlas_targets(rgba,texture['metallicRoughnessTexture'],material)
            uv,samples=uniform_uv_sample(rgba,texture['metallicRoughnessTexture'],material)
            native=ns['foreground_material_targets'](rgba)
            out=root/asset/'derived';out.mkdir()
            Image.fromarray(rgba).save(out/'input-atlas-RGBA.png')
            np.savez_compressed(out/'material-targets-private.npz',UV=uv,albedo=samples['albedo'].astype(np.float32),
                roughness=samples['roughness'].astype(np.float32),metallic=samples['metallic'].astype(np.float32),
                native_batch_albedo=targets['albedo'],native_batch_roughness=targets['roughness'],native_batch_metallic=targets['metallic'])
            delta={key:float(np.mean(np.abs(np.asarray(native[key])-targets[key]))) for key in ('albedo','roughness','metallic')}
            record.update(passed=True,source_legal=metadata['legal'],PNG_dimensions=[rgba.shape[1],rgba.shape[0]],
                input_RGBA_pixel_SHA256=pixelhash,actual_UV_slots=256,full_uniform_UV_texels=count,
                target_shapes={k:list(v.shape) for k,v in targets.items()},target_extrema=extrema,
                frozen_native_RGB_heuristic_vs_physical_UV_mean_abs_difference=delta,
                derived_files={p.name:pin(p.read_bytes()) for p in out.iterdir()},
                frozen_native_target_function_executed=True,bridge_semantics='Linear baseColor, linear G-roughness/B-metallic; uniform UV atlas labels, not observed rendered surface-area labels',
                future_cache_missing=['frozen triplane_features and image_features from actual native encoder','native rendered RGBA/camera/UV correspondence if changing atlas recipe to real object views','matching new A100 reference/numeric/update/recovery','third untouched object for blind group'])
            result['completed_objects']+=1;result['actual_UV_slots']+=256
        except Exception as e:
            record.update(error_type=type(e).__name__,reason=str(e)[:512]);result['failed_objects']+=1
        save(root,'progress.json',result)
    result.update(passed=result['completed_objects']==2,state='REAL_PBR_ATLAS_INPUT_TARGET_BRIDGE_ONLY' if result['completed_objects']==2 else 'FAILED_PARTIAL_PRESERVED',
        split_groups={'train':['WaterBottle'],'dev':['BoomBox'],'blind':[]},raw_and_decoded_cross_group_duplicates=0 if result['completed_objects']==2 else None,
        dataset_split_semantics='All original object textures/UV samples/crops/derived views stay in one object group; same Microsoft author/style is not independent-domain/OOD proof',
        planned_blind_objects=0,material_target_MAE_model_effect_measured=False,actual_frozen_encoder_invocations=0,
        numpy=importlib.metadata.version('numpy'),Pillow=importlib.metadata.version('Pillow'),
        source_branch_snapshot_consistency='same main URL family, per-body identities pinned; upstream full commit not resolved')
    save(root,'result.json',result)
    print(json.dumps({k:result[k] for k in ['passed','state','planned_objects','completed_objects','failed_objects','download_bytes','actual_UV_slots','READY']},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
