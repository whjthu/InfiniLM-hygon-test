"""Actual two accessible objects; preserve inaccessible third object, no retry."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from surface_targets import (RECIPE,N,need,pin,fixed,source_module,surface_samples,
                             load_native_targets,to_original_target_object,training_identity,sample_geometry)

D=Path(__file__).resolve().parent;G=D.parents[2]
OUT=G/'local-only/training-sf3d-public-surface-material-targets-v1-original'
OLD=G/'local-only/training-sf3d-public-pbr-atlas2-v1-original'
MESH=G/'local-only/training-trellis-public-mesh-quality-input-v1-original'
DEV=G/'local-only/training-sf3d-BoomBox-ORM-onefile-completion-v1-original'
SPECS={
 'WaterBottle':{'split':'train','gltf':OLD/'WaterBottle/glTF/WaterBottle.gltf','gltf_pin':{'bytes':2727,'SHA256':'0596f4e61dc781439d254fdfb5e3462daf1762c18715e3e3ac13001aa8f3f547'},
   'mesh':{'bytes':149412,'SHA256':'e4921f2d0c0a03cf65286bd195f3688d4f99d7b6a58f25935e70d516d744156e'},
   'base':OLD/'WaterBottle/glTF/WaterBottle_baseColor.png','base_pin':{'bytes':2165850,'SHA256':'62d0ad9b4e2c75e9bc7d67644a50d697563000ffa586d2e185ea38f5000e0a63'},
   'orm':OLD/'WaterBottle/glTF/WaterBottle_occlusionRoughnessMetallic.png','orm_pin':{'bytes':3577832,'SHA256':'10cc809d9762e8c671f1ae73618dd5fdd07792d2c6ec9c4691cdb991617c9369'},
   'old_target':OLD/'WaterBottle/derived/material-targets-private.npz','old_target_pin':{'bytes':2930,'SHA256':'309475d22858a0d2659519f27aba91a721045e069364f1b3c52674d46f623eea'}},
 'BoomBox':{'split':'dev','gltf':OLD/'BoomBox/glTF/BoomBox.gltf','gltf_pin':{'bytes':2702,'SHA256':'b1d2e0e56864c71a4b65fc560bfc1572b1eb32cfb9c8e86d06e6f9ace2803a43'},
   'mesh':{'bytes':207816,'SHA256':'f6c61a95af3e0e2462a9ac01d1e681706149452e0191930e0591a81d50b5b944'},
   'base':OLD/'BoomBox/glTF/BoomBox_baseColor.png','base_pin':{'bytes':3285844,'SHA256':'099816a7afc5f6690494313ac8039806fd6d5b84179126a808b2678aaab3563a'},
   'orm':DEV/'BoomBox_occlusionRoughnessMetallic.png','orm_pin':{'bytes':4471451,'SHA256':'496704a4836ff364dc4441f41651edb25178498926c859933e33df42d6361412'},
   'old_target':DEV/'dev-material-targets-private.npz','old_target_pin':{'bytes':2912,'SHA256':'70f10ea437e951a72295e30df989b166d298f79472700a9ebb1f78f8e4525b7d'}}}


def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def build_one(group,accessor,decode):
    x=SPECS[group];g=json.loads(fixed(x['gltf'],x['gltf_pin']));blob=fixed(MESH/(group+'.bin'),x['mesh'])
    primitive=g['meshes'][0]['primitives'][0];material=g['materials'][primitive['material']]['pbrMetallicRoughness']
    for key,expected in [('baseColorTexture',group+'_baseColor.png'),('metallicRoughnessTexture',group+'_occlusionRoughnessMetallic.png')]:
        tex=g['textures'][material[key]['index']];need(g['images'][tex['source']]['uri']==expected,'Material texture URI association differs')
    with Image.open(io.BytesIO(fixed(x['base'],x['base_pin']))) as im:base=np.asarray(im.convert('RGBA'))
    with Image.open(io.BytesIO(fixed(x['orm'],x['orm_pin']))) as im:orm=np.asarray(im.convert('RGB'))
    fields,stats=surface_samples(g,blob,accessor,base,orm,decode,group)
    path=OUT/(group+'-surface-material-targets-private.npz');np.savez_compressed(path,**fields)
    wanted=pin(path.read_bytes());native=load_native_targets(path,wanted,RECIPE);target=to_original_target_object(native)
    need(target['albedo'].shape==(3,) and type(target['roughness']) is float and type(target['metallic']) is float,'Original target-object argument ABI differs')
    with np.load(io.BytesIO(fixed(x['old_target'],x['old_target_pin'])),allow_pickle=False) as old:
        deltas={k:float(np.max(np.abs(native[k].astype(float)-old['native_batch_'+k].astype(float)))) for k in native}
    return {'group':group,'split':x['split'],'completed':True,'material_samples':N,'mesh':x['mesh'],
        'gltf':x['gltf_pin'],'textures':{'base':x['base_pin'],'orm':x['orm_pin']},'target':wanted,
        'source_surface':stats,'global_native_target':{k:a.tolist() for k,a in native.items()},
        'native_fields':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'finite_0to1':True} for k,a in native.items()},
        'compatible_original_target_object':True,'absolute_change_from_existing_atlas_average':deltas,
        'old_target':x['old_target_pin'],'old_target_files_modified':False,
        'triplane_or_image_features_or_model_predictions_created':False}


def main():
    started=time.monotonic();OUT.mkdir()
    save('intent.json',{'request':'f44a8a3f5b3747979e42e9ba3a13149e','recipe':RECIPE,'planned_groups':['WaterBottle','Corset','BoomBox'],
        'sample_count_per_group':N,'seed_rule':'SHA256(group + -surface-PBR4096-v1) first16hex','network_or_permission_requests':0,
        'known_Corset_source_PermissionError_not_retried':True,'native_backend_target_object_boundary':'sf3d_training.py:1110 and1127-1129'})
    r={'schema':'SF3D_REAL_PUBLIC_SURFACE_GLOBAL_MATERIAL_INPUT_LOCAL_V1','passed':False,'recipe':RECIPE,'planned_objects':3,'completed_objects':0,
       'failed_or_blocked_objects':1,'planned_surface_slots':3*N,'completed_surface_slots':0,
       'objects':{'Corset':{'group':'Corset','split':'train','completed':False,'state':'PRECHECK_PERMISSION_ERROR_NOT_RETRIED',
                   'exact_missing_read':'local-only/training-sf3d-public-corset-train-atlas-v1-original/Corset.gltf',
                   'expected':{'bytes':2487,'SHA256':'1aaa7fa54731ca833c3a0f919ba82c1af2a69cc7f8de36ba76bfb5038de8bd38'}}},
       'HTTP_SSH_Slurm_GPU_or_optimizer_updates':0,'model_or_feature_body_reads':0,'READY':False}
    try:
        mq=source_module('surface_existing_glTF_accessor',G/'code/bw1000/training-trellis-public-mesh-quality-input-v1/mesh_quality.py',
             {'bytes':5502,'SHA256':'8cd2e29e3708319e0ce161e350108b6e1217d43a0d48ceff8ed6d6f68e27456a'})
        mf=json.loads(fixed(G/'code/bw1000/training-sf3d-public-pbr-atlas2-v1/SOURCE_FILES.json',{'bytes':986,'SHA256':'9918612f5815bd99fad10321f20735a20a36cf9888adaa2979db04ca196a3b0f'}))
        pbr=source_module('surface_existing_PBR_decoder',G/'code/bw1000/training-sf3d-public-pbr-atlas2-v1/pbr.py',mf['files']['pbr.py'])
        for group in ['WaterBottle','BoomBox']:
            try:
                need(time.monotonic()-started<60,'Bounded local material-input epoch elapsed')
                item=build_one(group,mq.accessor,pbr.decode_pixels);r['objects'][group]=item
                r['completed_objects']+=1;r['completed_surface_slots']+=N
            except Exception as e:
                r['objects'][group]={'group':group,'split':SPECS[group]['split'],'completed':False,'error_type':type(e).__name__,'reason':str(e)[:300]}
                r['failed_or_blocked_objects']+=1
        # Independent geometric sampling check: triangle areas1:4, no material
        # labels/old outputs involved; only sampling coverage is tested here.
        v=np.array([[0,0,0],[2,0,0],[0,1,0],[10,0,0],[14,0,0],[10,2,0]],float)
        f=np.array([[0,1,2],[3,4,5]],int);uv=np.zeros((6,2))
        _,chosen,_,_=sample_geometry(v,f,uv,20261121,count=8192)
        ratio=float((chosen==1).mean());need(abs(ratio-.8)<.02,'Area-weighted sampling geometric oracle failed')
        rejected=False
        try:training_identity(list(r['objects'].values()))
        except ValueError:rejected=True
        need(rejected,'One accessible train object must not pass two-object identity')
        r.update(local_sampling_oracle={'passed':True,'independent_triangle_area_ratio':[1,4],'planned_samples':8192,'observed_large_triangle_fraction':ratio},
           incomplete_train_identity_rejected=True,completed_train_objects=sum(x.get('completed',False) and x['split']=='train' for x in r['objects'].values()),
           completed_dev_objects=sum(x.get('completed',False) and x['split']=='dev' for x in r['objects'].values()),
           blind_objects_read=0,Torch_imported=any(n=='torch' or n.startswith('torch.') for n in sys.modules),
           source_groups_cross_split_overlap=0,state='REAL_TWO_SURFACE_LABELS_NATIVE_SHAPES_PASS_COHORT_BLOCKED_ONE_SOURCE',
           native_cache_features_or_resume_qualified=False)
    except Exception as e:r.update(error_type=type(e).__name__,reason=str(e)[:512],state='FAILED_OR_PARTIAL')
    r['wall_seconds']=time.monotonic()-started;save('result.json',r)
    print(json.dumps({k:r[k] for k in ['state','planned_objects','completed_objects','failed_or_blocked_objects','completed_surface_slots','READY']}))
    return 0 if r['completed_objects']==2 and r.get('incomplete_train_identity_rejected') else 1


if __name__=='__main__':raise SystemExit(main())
