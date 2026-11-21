"""One exact known-size ORM GET; immutable old failed scope and training targets."""
import argparse
import ast
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.request

D=Path(__file__).resolve().parent
G=D.parents[2]
OLD=G/'local-only/training-sf3d-public-pbr-atlas2-v1-original'
FROZEN=G/'code/bw1000/training-sf3d-public-pbr-atlas2-v1'
URL='https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoomBox/glTF/BoomBox_occlusionRoughnessMetallic.png'
EXPECTED_BYTES=4471451
EXPECTED_ETAG='"1e23b18434bb370e5ff5240aa2ef875862cfe971c4576669b85abe6a85d72c6b"'


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(raw):return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def fixed(path,size,sha):
    raw=path.read_bytes();need(pin(raw)=={'bytes':size,'SHA256':sha},'frozen bytes differ: '+path.name);return raw


def save(root,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with (root/name).open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(raw)


def headers_gate(status,headers):
    need(status==200,'full exact HTTP200 required')
    need(headers.get_all('Content-Length')==[str(EXPECTED_BYTES)],'exact observed Content-Length required before body')
    need(headers.get_all('ETag')==[EXPECTED_ETAG],'exact single original ETag required before body')
    need(headers.get('Content-Type','').split(';')[0].strip().lower()=='image/png','actual PNG framing required')
    need(headers.get_all('Content-Encoding') in (None,['identity']),'encoded body forbidden')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline',required=True,type=float);a=p.parse_args();root=Path(a.root)
    while not (root/'job-gate.json').exists():
        need(time.time()<a.deadline,'actual Job gate expired');time.sleep(.01)
    gate=json.loads((root/'job-gate.json').read_bytes());need(gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648,'actual Job gate differs')
    result={'schema':'SF3D_BOOMBOX_EXACT_ORM_COMPLETION_V1','passed':False,'phase':'original-prerequisites','planned_new_GETs':1,
        'actual_new_GETs':0,'actual_new_body_bytes':0,'new_body_natural_EOF':False,'headers_qualified':False,
        'planned_objects':2,'completed_objects':0,'planned_UV_slots':512,'completed_UV_slots':0,
        'original_failed_scope_preserved':True,'blind_object_groups':0,'actual_GPU_model_encoder_training':0,
        'full_native_cache_qualified':False,'READY':False}
    try:
        old_result=json.loads(fixed(OLD/'result.json',9423,'d5d9a18acd4aecff3e5ba12dfb1cefbfea76c1e1b09447d1d81473909c80fa48'))
        old_out=json.loads(fixed(OLD/'outcome.json',1075,'c663dec51feaf42944709209b803b679a0dc06e6513706c3785d96a2a182f3e1'))
        public=G/'results/bw1000/2026-11-21/training-sf3d-public-PBR-atlas-real-input-and-label-bridge-partial-local.json'
        fixed(public,8130,'3d7fbdd95a14350a5d21340bab80d175186a363a5aa10dbe19a02268050948ee')
        need(old_out['state']=='KNOWN_TERMINAL' and old_out['passed'] is False and old_out['Job']['returncode']==1 and len(old_result['requests'])==10,'old known failure must remain unchanged')
        original=old_result['requests'][-1]
        need(original['URL']==URL and original['status']==200 and original['ETag']==EXPECTED_ETAG and original['Content_Length']==str(EXPECTED_BYTES)
             and original['read_bytes']==original['saved_bytes']==0 and original['EOF'] is False,'original zero-body exact identity not established')
        manifest=json.loads(fixed(FROZEN/'SOURCE_FILES.json',986,'9918612f5815bd99fad10321f20735a20a36cf9888adaa2979db04ca196a3b0f'))
        for name,q in manifest['files'].items():fixed(FROZEN/name,q['bytes'],q['SHA256'])
        gltf_raw=fixed(OLD/'BoomBox/glTF/BoomBox.gltf',2702,'b1d2e0e56864c71a4b65fc560bfc1572b1eb32cfb9c8e86d06e6f9ace2803a43')
        gltf=json.loads(gltf_raw);material=gltf['materials'][0];pbr=material['pbrMetallicRoughness']
        need(gltf['images'][gltf['textures'][pbr['metallicRoughnessTexture']['index']]['source']]['uri']=='BoomBox_occlusionRoughnessMetallic.png','frozen glTF target differs')
        metadata=json.loads(fixed(OLD/'BoomBox/metadata.json',655,'6fdf2571e662907ba51b35ca303d616baca22ae5f8e689bfd2110ef06329236a'))
        fixed(OLD/'BoomBox/LICENSE.md',770,'0fbb151acb88aadda72f05c54c05429bfae2c8af0b0f1ddc0c636be3a55da4c0')
        need(any(x['license']=='CC0-1.0' and x['artist']=='Microsoft' and x['what']=='Everything' for x in metadata['legal']),'original asset rights differ')
        result['phase']='one-new-exact-ORM-GET';result['actual_new_GETs']=1
        save(root,'progress.json',result)
        req=urllib.request.Request(URL,headers={'Accept-Encoding':'identity','If-Match':EXPECTED_ETAG})
        need(time.time()<a.deadline,'original epoch exhausted before GET')
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=min(30,a.deadline-time.time())) as response:
            result['response_headers']={'status':response.status,'Content_Length':response.headers.get_all('Content-Length'),'ETag':response.headers.get_all('ETag'),
                'Content_Type':response.headers.get('Content-Type'),'Content_Encoding':response.headers.get_all('Content-Encoding')}
            headers_gate(response.status,response.headers);result['headers_qualified']=True
            sha=hashlib.sha256()
            with (root/'BoomBox_occlusionRoughnessMetallic.png').open('xb') as f:
                while True:
                    need(time.time()<a.deadline,'same original completion epoch exhausted')
                    b=response.read1(min(65536,EXPECTED_BYTES+1-result['actual_new_body_bytes']))
                    if not b:result['new_body_natural_EOF']=True;break
                    result['actual_new_body_bytes']+=len(b)
                    need(result['actual_new_body_bytes']<=EXPECTED_BYTES,'body larger than exact observed size')
                    f.write(b);sha.update(b)
                f.flush();os.fsync(f.fileno())
            need(result['new_body_natural_EOF'] and result['actual_new_body_bytes']==EXPECTED_BYTES,'complete exact body/EOF required')
            result['new_ORM']={'bytes':EXPECTED_BYTES,'SHA256':sha.hexdigest()}
        result['phase']='reuse-original-inputs-and-PBR-math'
        import numpy as np
        from PIL import Image
        spec=importlib.util.spec_from_file_location('published_frozen_pbr',FROZEN/'pbr.py');math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
        src=(OLD/'BoomBox/glTF/BoomBox_baseColor.png')
        base=fixed(src,3285844,'099816a7afc5f6690494313ac8039806fd6d5b84179126a808b2678aaab3563a')
        newraw=(root/'BoomBox_occlusionRoughnessMetallic.png').read_bytes();need(pin(newraw)==result['new_ORM'],'downloaded file changed')
        def decode(raw,mode):
            with Image.open(BytesIO(raw)) as image:
                need(image.format=='PNG' and 0<image.width<=4096 and 0<image.height<=4096,'bounded actual PNG required');image.load()
                return np.asarray(image.convert(mode)).copy()
        rgba=decode(base,'RGBA');need(material.get('alphaMode','OPAQUE')=='OPAQUE','frozen opaque atlas scope differs');rgba[...,3]=255
        orm=decode(newraw,'RGB');targets,extrema,texels=math.atlas_targets(rgba,orm,material);uv,samples=math.uniform_uv_sample(rgba,orm,material)
        archive=G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
        fixed(archive,738802,'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f')
        with tarfile.open(archive,'r:gz') as t:body=t.extractfile('framework/generation/src/gencuda/backends/sf3d_training.py').read()
        need(pin(body)=={'bytes':58458,'SHA256':'dc57683c87f77c62635358f8a65db30c3070f000d235a60ad28f9395e4b68802'},'frozen backend differs')
        function=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='foreground_material_targets')
        ns={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen-original-SF3D-target-function','exec'),ns)
        native=ns['foreground_material_targets'](rgba)
        Image.fromarray(rgba).save(root/'dev-input-atlas-RGBA.png')
        np.savez_compressed(root/'dev-material-targets-private.npz',UV=uv,albedo=samples['albedo'].astype(np.float32),roughness=samples['roughness'].astype(np.float32),
            metallic=samples['metallic'].astype(np.float32),native_batch_albedo=targets['albedo'],native_batch_roughness=targets['roughness'],native_batch_metallic=targets['metallic'])
        result['phase']='two-object-cross-split-pixel-dedup-and-target-read'
        train_target=fixed(OLD/'WaterBottle/derived/material-targets-private.npz',2930,'309475d22858a0d2659519f27aba91a721045e069364f1b3c52674d46f623eea')
        fixed(OLD/'WaterBottle/derived/input-atlas-RGBA.png',1931594,'46317ecf00b8dcc41b781b916379a0d2b0b7cacfc8dd7f0da8819e6861116ead')
        train_base=fixed(OLD/'WaterBottle/glTF/WaterBottle_baseColor.png',2165850,'62d0ad9b4e2c75e9bc7d67644a50d697563000ffa586d2e185ea38f5000e0a63')
        train_orm=fixed(OLD/'WaterBottle/glTF/WaterBottle_occlusionRoughnessMetallic.png',3577832,'10cc809d9762e8c671f1ae73618dd5fdd07792d2c6ec9c4691cdb991617c9369')
        ta=decode(train_base,'RGBA');ta[...,3]=255;tr=decode(train_orm,'RGB')
        def pixels(a):
            if a.shape[-1]==3:a=np.concatenate([a,np.full(a.shape[:2]+(1,),255,np.uint8)],axis=-1)
            return {'shape':list(a.shape),'RGBA_pixels_SHA256':hashlib.sha256(a.tobytes()).hexdigest()}
        pixel_pins={'train-baseColor':pixels(ta),'train-ORM':pixels(tr),'dev-baseColor':pixels(rgba),'dev-ORM':pixels(orm)}
        need(pixel_pins['train-baseColor']['RGBA_pixels_SHA256']==old_result['objects']['WaterBottle']['input_RGBA_pixel_SHA256'],'frozen training input pixels changed')
        need(len({p['RGBA_pixels_SHA256'] for p in pixel_pins.values()})==4,'complete four-texture decoded pixel collision')
        raw_pins=[pin(train_base),pin(train_orm),pin(base),result['new_ORM']]
        need(len({p['SHA256'] for p in raw_pins})==4,'complete raw content collision')
        spec=importlib.util.spec_from_file_location('published_target_bridge',FROZEN/'read_targets.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        train=bridge.load_targets(OLD/'WaterBottle/derived/material-targets-private.npz',pin(train_target),'sf3d-public-pbr-UV-atlas-v1')
        dev_pin=pin((root/'dev-material-targets-private.npz').read_bytes())
        dev=bridge.load_targets(root/'dev-material-targets-private.npz',dev_pin,'sf3d-public-pbr-UV-atlas-v1')
        with np.load(BytesIO(train_target),allow_pickle=False) as z:
            need(z['UV'].shape==(256,2) and np.array_equal(z['UV'],uv),'same fixed256 UV coordinates required')
            need(np.isfinite(z['albedo']).all() and np.isfinite(z['roughness']).all() and np.isfinite(z['metallic']).all(),'frozen training labels nonfinite')
        delta={k:float(np.mean(np.abs(np.asarray(native[k])-dev[k]))) for k in targets}
        artifacts={name:pin((root/name).read_bytes()) for name in ['BoomBox_occlusionRoughnessMetallic.png','dev-input-atlas-RGBA.png','dev-material-targets-private.npz']}
        binding={'schema':'SF3D_PUBLIC_PBR_ATLAS2_COMPLETE_ASSET_BINDING_V1','objects':{'train':{'group':'WaterBottle','input':str((OLD/'WaterBottle/derived/input-atlas-RGBA.png').relative_to(G)),
            'target':str((OLD/'WaterBottle/derived/material-targets-private.npz').relative_to(G)),'input_pin':old_result['objects']['WaterBottle']['derived_files']['input-atlas-RGBA.png'],'target_pin':pin(train_target)},
            'dev':{'group':'BoomBox','input':str((root/'dev-input-atlas-RGBA.png').relative_to(G)),'target':str((root/'dev-material-targets-private.npz').relative_to(G)),'input_pin':artifacts['dev-input-atlas-RGBA.png'],'target_pin':dev_pin}},
            'blind_groups':[],'four_texture_pixel_pins':pixel_pins,'original_failure_preserved':True,'training_or_model_READY':False}
        binding_pin=save(root,'complete-data-binding-private.json',binding)
        result.update(passed=True,state='TWO_PUBLIC_PBR_OBJECTS_ASSETS_COMPLETE_NOT_NATIVE_TRAINING',phase='complete',completed_objects=2,completed_UV_slots=512,
            dev_PNG_dimensions=[rgba.shape[1],rgba.shape[0]],dev_uniform_UV_texels=texels,dev_target_extrema=extrema,
            exact_content_ETag_reused=True,complete_four_raw_unique=4,complete_four_decoded_RGBA_unique=4,cross_object_raw_or_exact_pixel_collisions=0,
            source_group_counts={'train':1,'dev':1,'blind':0},native_target_shapes={group:{k:list(v.shape) for k,v in fields.items()} for group,fields in [('train',train),('dev',dev)]},
            old_training_target_and_input_immutable=True,train_native_heuristic_not_reexecuted=True,
            dev_native_foreground_function_executions=1,dev_heuristic_vs_PBR_UV_mean_abs_difference=delta,
            assets=artifacts,complete_binding=binding_pin,actual_cumulative_GETs=11,actual_cumulative_received_bytes=9037799+EXPECTED_BYTES,
            source_identity='original official main per-file snapshots plus same strong ETag for new body; upstream commit still UNKNOWN',
            measured_PBR_model_MAE_or_camera_LPIPS=False,semantic_or_near_duplicate_OOD_proof=False)
    except urllib.error.HTTPError as e:
        result.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,HTTP_status=e.code);e.close()
    except Exception as e:
        import traceback
        save(root,'failure-private.json',{'phase':result['phase'],'type':type(e).__name__,'reason':str(e)[:1024],'trace':traceback.format_exc()[-4096:]})
        result.update(state='FAILED_PARTIAL_PRESERVED',error_type=type(e).__name__,failure=pin((root/'failure-private.json').read_bytes()))
    save(root,'result.json',result)
    print(json.dumps({k:result[k] for k in ['passed','state','actual_new_GETs','actual_new_body_bytes','completed_objects','completed_UV_slots','READY']},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
