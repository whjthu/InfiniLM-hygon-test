"""Existing original20 fingerprints and exact native routing, never native media.

No Torch/pandas imports, weights, image bodies, old captures or network.
The fingerprint record exercise is explicitly not CSV/PIL preprocessing qualification.
"""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

G=Path(__file__).resolve().parents[3]
REQUEST='3fcf9dd3252a4278894b9ca3f8df12d5'


def checked(path,size,digest):
    raw=path.read_bytes()
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('original source or metadata identity differs')
    return raw


def main():
    start=time.monotonic()
    out=G/'local-only/training-zimage-original20-input-contract-CPU-v1-original'
    out.mkdir(exist_ok=False)
    base=G/'code/bw1000/training-zimage-A100-native-first-updates-v1'
    raw=checked(base/'BINDING_IDENTITY.json',120160,'a3d62f6cc61ba8d472d61dc7394c262fa167187d577c6c27f28a7bf3930f3ae1')
    original=json.loads(raw)
    registered=json.loads((base/'SOURCE_FILES.json').read_bytes())['files']['BINDING_IDENTITY.json']
    assert registered==dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    config,data=original['config'],original['data']
    assert original['case']=='T07.sft' and config['height']==config['width']==256
    assert config['lora_rank']==4 and config['max_steps']==25 and config['warmup_steps']==5
    assert config['lora_target_modules']=='to_q,to_k,to_v,to_out.0,w1,w2,w3'
    assert data['manifest']==dict(bytes=1932,SHA256=config['dataset_sha'])
    assert data['metadata']==dict(bytes=1521,SHA256='6513edae65017e75494e9b49d644f8f4d46877cddf1c8acf2ddfb33c83f2816e')
    order=data['row_order'];prompts=data['row_prompt_SHA256'];images=data['images']
    assert len(order)==len(set(order))==len(prompts)==len(images)==20
    assert set(order)==set(images) and len(set(prompts))==20
    assert len({v['SHA256'] for v in images.values()})==20
    assert sum(v['bytes'] for v in images.values())==3644668
    for name in order:
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and p.as_posix()==name and p.parts[0]=='images'
    native_root=G/'code/bw1000/training-zimage-target-native-meta-CPU-v1/upstream'
    native=checked(native_root/'diffsynth/core/data/unified_dataset.py',4972,'b6e0729fbd00deee8b0d03dd463df43ce7e653ecea3f75c18741d4cc03ac027f')
    operators=checked(native_root/'diffsynth/core/data/operators.py',7735,'c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4')
    assert original['source_files']['diffsynth/core/data/unified_dataset.py']['SHA256']==hashlib.sha256(native).hexdigest()
    assert original['source_files']['diffsynth/core/data/operators.py']['SHA256']==hashlib.sha256(operators).hexdigest()
    cls=next(n for n in ast.parse(native).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedDataset')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'__getitem__','__len__'}]
    assert len(methods)==2
    # Keep both native method bodies unchanged; object base avoids importing Torch.
    thin=ast.ClassDef(name='NativeFingerprintRouting',bases=[],keywords=[],body=methods,decorator_list=[])
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[thin],type_ignores=[])),'pinned-native-routing','exec'),ns)
    dataset=ns['NativeFingerprintRouting']()
    dataset.load_from_cache=False;dataset.max_data_items=None;dataset.repeat=1
    dataset.data_file_keys=();dataset.special_operator_map={}
    dataset.data=[dict(image=name,prompt_SHA256=prompts[i]) for i,name in enumerate(order)]
    assert len(dataset)==20
    archive=checked(G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz',738802,'85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f')
    import io
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        backend=tar.extractfile('framework/generation/src/gencuda/backends/zimage_training.py').read()
    assert len(backend)==31044 and hashlib.sha256(backend).hexdigest()=='6526be513a7b55977ede0dd176757da1530fffedaefa0080639e4191b1f49d31'
    run=next(n for n in ast.parse(backend).body if isinstance(n,ast.FunctionDef) and n.name=='run_zimage_training')
    assignment=next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sample_index' for t in n.targets))
    assignment_code=compile(ast.Module(body=[assignment],type_ignores=[]),'pinned-original-sample-index','exec')
    def route(step):
        scope=dict(global_step=step,dataset=dataset);exec(assignment_code,scope)
        index=scope['sample_index'];row=dataset[index]
        assert row==dict(image=order[index],prompt_SHA256=prompts[index])
        return index
    continuous=[route(step) for step in range(1,26)]
    resumed=[route(step) for step in range(21,26)]
    assert resumed==continuous[20:] and continuous==list(range(20))+list(range(5))
    counts=[continuous.count(i) for i in range(20)]
    assert counts==[2]*5+[1]*15
    # The original CSV and media are on A100. Do not reconstruct captions from hashes.
    known_csv_dump=base/'original-inputs.stdout.private.json'
    known_capture=G/'local-only/training-zimage-A100-native-first-updates-v1/original-inputs.stdout.private.json'
    result=dict(schema='ZIMAGE_ORIGINAL20_FROZEN_INPUT_CONTRACT_CPU_V1',passed=True,request_id=REQUEST,
        original_binding=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()),
        original_unique_rows=20,original_unique_image_body_SHA=20,original_unique_prompt_SHA=20,
        declared_image_bytes=3644668,native_provider_bodies_SHA_matched=True,
        exact_native_routing_methods=['UnifiedDataset.__getitem__','UnifiedDataset.__len__'],
        exact_backend_sample_index_AST=True,continuous_optimizer_slots=25,resume20_tail_slots=5,
        sample_cursor_after_10=10,sample_cursor_after_20=0,sample_cursor_after_25=5,
        occurrences_per_original=[2]*5+[1]*15,fingerprint_routing_only=True,
        known_local_raw_dump_exists=known_csv_dump.is_file(),known_local_capture_exists=known_capture.is_file(),
        real_CSV_parsed=False,real_image_decoded=False,pandas_or_Torch_imported=False,
        image_preprocess='Original native source: RGB, torchvision BILINEAR resize, center_crop256; not executed here',
        expected_CSV_columns=['image','prompt'],SDXL_text_imagefolder_interchangeable=False,
        official_split_or_dedup_beyond_original20_qualified=False,
        actual_optimizer_updates=0,HTTP=0,SSH=0,GPU=0,weights_body_read=0,
        target_READY=False,training_READY_increment=0,wall_seconds=time.monotonic()-start,
        remaining_exact_original_files={
          'metadata':dict(path=config['dataset_metadata'],**data['metadata']),
          'manifest':dict(path=config['dataset_manifest'],**data['manifest']),
          'media':dict(root=data['root'],files=20,bytes=3644668,each_file_pin_in_original_binding=True)},
        limitations=['This is real frozen metadata/source contract validation, not actual COCO20 image/CSV processing',
                     'Original media rights/attribution not requalified or expanded',
                     'Preset fusion, HIP attention/device and native full update/recovery remain independent prerequisites'])
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['passed','request_id','original_unique_rows','continuous_optimizer_slots','resume20_tail_slots','real_CSV_parsed','real_image_decoded','actual_optimizer_updates','training_READY_increment']}))


if __name__=='__main__':main()
