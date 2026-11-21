"""New official mesh buffers only. No texture GET, checkpoint or model operation."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request
import urllib.error

D=Path(__file__).resolve().parent
G=D.parents[2]
PRIOR=G/'local-only/training-sf3d-public-pbr-atlas2-v1-original'
CORSET=G/'local-only/training-sf3d-public-corset-train-atlas-v1-original'
SPECS=[('WaterBottle','train',PRIOR/'WaterBottle/glTF/WaterBottle.gltf',2727,'0596f4e61dc781439d254fdfb5e3462daf1762c18715e3e3ac13001aa8f3f547',149412),
    ('Corset','train',CORSET/'Corset.gltf',2487,'1aaa7fa54731ca833c3a0f919ba82c1af2a69cc7f8de36ba76bfb5038de8bd38',662184),
    ('BoomBox','dev',PRIOR/'BoomBox/glTF/BoomBox.gltf',2702,'b1d2e0e56864c71a4b65fc560bfc1572b1eb32cfb9c8e86d06e6f9ace2803a43',207816)]


def pin(b):return {'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest()}


def need(ok,why):
    if not ok:raise ValueError(why)


def fixed(path,size,sha):
    raw=path.read_bytes();need(pin(raw)=={'bytes':size,'SHA256':sha},'exact source identity differs: '+path.name);return raw


def save(root,name,value):
    with (root/name).open('w',encoding='utf8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline',required=True,type=float);a=p.parse_args();root=Path(a.root)
    while not (root/'job-gate.json').exists():need(time.time()<a.deadline,'Job gate deadline');time.sleep(.01)
    gate=json.loads((root/'job-gate.json').read_bytes());need(gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648,'real Job differs')
    import numpy as np
    from mesh_quality import read_geometry,sample_surface,point_metrics
    result={'schema':'TRELLIS_PUBLIC_REFERENCE_GEOMETRY_INPUT_V1','passed':False,'planned_objects':3,'completed_objects':0,'failed_objects':0,
        'planned_GETs':3,'actual_GETs':[],'download_bytes':0,'planned_reference_points':6144,'completed_reference_points':0,
        'actual_model_or_GPU_or_optimizer_or_latent_encoders':0,'READY':False,'objects':{},'blind_groups':0}
    archive=G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
    fixed(archive,738802,'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f')
    with tarfile.open(archive,'r:gz') as t:body=t.extractfile('framework/generation/src/gencuda/trellis_cpu_texture_export.py').read()
    need(pin(body)=={'bytes':43773,'SHA256':'f8fbbe622e6a792824b532906ae9d2c19405005519e1f2169d7fb0d97af35e85'},'native exporter source differs')
    functions=[n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name in ('require','corner_geometry')]
    ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])),'frozen-native-trellis-geometry-functions','exec'),ns)
    result['native_source']={'member':'generation/src/gencuda/trellis_cpu_texture_export.py',**pin(body),'function':'corner_geometry','functions_AST_SHA256':{n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in functions},'whole_module_Torch_or_trimesh_imported':False}
    opener=urllib.request.build_opener(NoRedirect())
    for asset,split,path,gs,gh,length in SPECS:
        item={'split':split,'passed':False,'source_declared_geometry_bytes':length};result['objects'][asset]=item
        try:
            gltf=json.loads(fixed(path,gs,gh));need(gltf['buffers']==[{'byteLength':length,'uri':asset+'.bin'}],'exact frozen buffer declaration')
            # Known original texture-only producer never requested these paths.
            need(not (path.parent/(asset+'.bin')).exists(),'Known geometry path already exists: reconcile instead of duplicate GET')
            url='https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/'+asset+'/glTF/'+asset+'.bin'
            record={'URL':url,'status':None,'read_bytes':0,'EOF':False};result['actual_GETs'].append(record);save(root,'progress.json',result)
            need(time.time()<a.deadline,'single epoch elapsed before original GET')
            with opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=min(20,a.deadline-time.time())) as response:
                record.update(status=response.status,Content_Length=response.headers.get_all('Content-Length'),ETag=response.headers.get_all('ETag'))
                need(response.status==200 and response.url==url and response.headers.get_all('Content-Length')==[str(length)] and response.headers.get_all('Content-Encoding') in (None,['identity']),'exact declared full200 source body required')
                need(record['ETag'] is not None and len(record['ETag'])==1,'one original source ETag required')
                with (root/(asset+'.bin')).open('xb') as f:
                    while True:
                        need(time.time()<a.deadline,'single original epoch exhausted')
                        b=response.read1(min(65536,length+1-record['read_bytes']))
                        if not b:record['EOF']=True;break
                        record['read_bytes']+=len(b);result['download_bytes']+=len(b)
                        need(record['read_bytes']<=length and result['download_bytes']<=1019412,'exact source/aggregate cap')
                        f.write(b)
                    f.flush();os.fsync(f.fileno())
                raw=(root/(asset+'.bin')).read_bytes();need(record['EOF'] and len(raw)==length,'exact EOF/length incomplete');record.update(pin(raw))
            vertices,faces,frame=read_geometry(gltf,raw)
            corners,normals,topology=ns['corner_geometry'](vertices,faces)
            seed=int(hashlib.sha256((asset+'-geometry-ref-v1').encode()).hexdigest()[:16],16)
            points=sample_surface(vertices,faces,seed)
            identity=point_metrics(points,points);far=point_metrics(points+np.array([2,0,0],np.float32),points)
            invalid=points.copy();invalid[0,0]=np.nan;failed=point_metrics(invalid,points)
            need(identity['Chamfer_L1']==0 and identity['Fscore_at_1percent']==1 and far['Fscore_at_1percent']==0 and far['Chamfer_L1']>=1 and not failed['passed'],'geometric metric diagnostic controls failed')
            output=root/(asset+'-reference-geometry-private.npz');np.savez_compressed(output,vertices=vertices,faces=faces,reference_points=points,
                center_m=np.array(frame['reference_center_m']),longest_extent_m=np.asarray(frame['reference_longest_extent_m']))
            item.update(passed=True,vertices=len(vertices),faces=len(faces),native_triangle_corners=len(corners),native_geometry=topology,
                frame=frame,reference_point_count=2048,sampling_seed=seed,sampling='triangle area weighted barycentric sqrt rule; no answer/model selection',
                geometry=pin(output.read_bytes()),metric_controls={'role':'diagnostic self/translated/invalid real mesh controls; NOT model quality','identity':identity,'disjoint_translation':far,'invalid_prediction':failed},
                source_gltf={'bytes':gs,'SHA256':gh},binary=record.copy(),artifact_role='PUBLIC_REFERENCE_MESH_NOT_MODEL_OUTPUT',native_full_export_or_ovoxel_or_shape_flow_qualified=False)
            result['completed_objects']+=1;result['completed_reference_points']+=2048
        except Exception as e:
            item.update(error_type=type(e).__name__,reason=str(e)[:512]);result['failed_objects']+=1
        save(root,'progress.json',result)
    result.update(passed=result['completed_objects']==3,state='REAL_THREE_OBJECT_NATIVE_GEOMETRY_GATE_AND_SCORE_INPUTS_PASS' if result['completed_objects']==3 else 'FAILED_PARTIAL_PRESERVED',
        exact_distinct_raw_buffers=len({x.get('SHA256') for x in result['actual_GETs'] if x.get('SHA256')}),
        group_split={'train':['WaterBottle','Corset'],'dev':['BoomBox'],'blind':[]},native_T10_generated_cache_or_original_A100_recipe_modified=False,
        actual_model_prediction_scores=0,near_duplicate_or_OOD_independence_proven=False,
        source_revision='Existing official main glTF snapshots plus newly received declared buffers by full bodySHA; upstream full commit UNKNOWN')
    save(root,'result.json',result)
    print(json.dumps({k:result[k] for k in ['passed','state','planned_objects','completed_objects','failed_objects','download_bytes','completed_reference_points','READY']},sort_keys=True))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
