"""Read the new T07 CSVs using exact common-backend evidence/routing functions."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import time
from typing import Any

G=Path(__file__).resolve().parents[3]
DATA=G/'local-only/training-zimage-USGS-style18-dataset-v1-original/dataset'
OUT=G/'local-only/training-zimage-USGS-style18-canonical-bridge-v1-original'


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def main():
    origin=time.monotonic();OUT.mkdir(exist_ok=False)
    archive=G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz'
    assert pin(archive.read_bytes())==dict(bytes=738802,SHA256='85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f')
    with tarfile.open(archive,'r:gz') as t:
        backend=t.extractfile('framework/generation/src/gencuda/backends/zimage_training.py').read()
        manifest=t.extractfile('framework/generation/src/gencuda/manifest.py').read()
    assert pin(backend)==dict(bytes=31044,SHA256='6526be513a7b55977ede0dd176757da1530fffedaefa0080639e4191b1f49d31')
    tree=ast.parse(backend)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_reject_symlink_components','_dataset_resume_evidence')]
    helper=next(n for n in ast.parse(manifest).body if isinstance(n,ast.FunctionDef) and n.name=='sha256_path')
    ns=dict(Path=Path,Any=Any,csv=csv,hashlib=hashlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper]+funcs,type_ignores=[])),'pinned-common-T07-functions','exec'),ns)
    binding_raw=(DATA/'data-binding-private.json').read_bytes();binding=json.loads(binding_raw)
    bysplit={s:[r for r in binding['images'] if r['split']==s] for s in ('train','dev')}
    observed={};all_images=set()
    for s,count in (('train',12),('dev',6)):
        root=DATA/s; raw=(root/'metadata.csv').read_bytes()
        assert pin(raw)==binding['metadata'][s]
        native,order=ns['_dataset_resume_evidence'](root,root/'metadata.csv')
        assert native['rows']==count and len(native['images'])==len(order)==count
        assert native['metadata_sha256']==pin(raw)['SHA256']
        expected=[{'path':Path(r['dataset_relative']).relative_to(s).as_posix(),'sha256':r['source']['SHA256']} for r in bysplit[s]]
        assert native['images']==expected
        assert order==[f"{native['metadata_sha256']}:{i}" for i in range(count)]
        hashes={r['sha256'] for r in native['images']}
        assert len(hashes)==count and not hashes&all_images;all_images.update(hashes)
        observed[s]=dict(rows=count,metadata=pin(raw),canonical_evidence_SHA256=pin(json.dumps(native,sort_keys=True).encode())['SHA256'],sample_order_SHA256=pin(json.dumps(order,separators=(',',':')).encode())['SHA256'])
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_zimage_training')
    assignment=next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='sample_index' for x in n.targets))
    route=compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),'canonical-step-routing','exec')
    def index(step):
        env=dict(global_step=step,dataset=range(12));exec(route,env);return env['sample_index']
    continuous=[index(i) for i in range(1,26)];resumed=[index(i) for i in range(21,26)]
    assert continuous==list(range(12))*2+[0] and resumed==continuous[20:]==[8,9,10,11,0]
    overlay=json.loads((DATA/'training-data-overlay.json').read_bytes())
    assert set(overlay)=={'dataset_root','dataset_metadata','dataset_manifest','dataset_sha'}
    assert overlay['dataset_root'].endswith('/dataset/train') and '/dev/' not in overlay['dataset_metadata']
    assert overlay['dataset_sha']==pin((DATA/'train/manifest.json').read_bytes())['SHA256']
    result=dict(schema='ZIMAGE_STYLE18_COMMON_BACKEND_CSV_EVIDENCE_LOCAL_V1',passed=True,
        actual_splits=observed,unique_image_SHA256=len(all_images),binding=pin(binding_raw),
        common_backend=pin(backend),common_manifest_helper_source=pin(manifest),
        exact_native_functions_executed=['_reject_symlink_components','_dataset_resume_evidence','sha256_path','run_zimage_training sample_index assignment'],
        original_image_bodies_rehashed=18,continuous25_slot_counts=[continuous.count(i) for i in range(12)],
        checkpoint20_next_row=8,resume20_to25_row_indices=resumed,dev_training_slots=0,
        actual_optimizer_updates=0,actual_GPU=0,actual_HTTP=0,Torch_imported=False,
        native_image_resize=False,actual_training_executed=False,target_READY=False,
        wall_seconds=time.monotonic()-origin)
    raw=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    (OUT/'result.json').write_bytes(raw)
    print(json.dumps(dict(passed=True,train=12,dev=6,rehash=18,checkpoint20_next_row=8,
                         actual_optimizer_updates=0,target_READY=False,wall_seconds=result['wall_seconds']),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
