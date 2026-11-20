"""Audit real frozen cache-producer semantics; no data/weights/device execution."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

G=Path(__file__).resolve().parents[3]
REQUEST='155777df74bf4fe8944699fd463d902da'
SPECS={
 'generation/scripts/prepare_hy_motion_training_cache.py':(16055,'7a8fceef0d4ba30c6676881d3c7dc59fed4c73b54e292d14b57bcdba1c40a203'),
 'generation/src/gencuda/backends/hy_motion_training.py':(49094,'e23a88e0b78ec16ef76e7c4eb0b254452f0c21a4013f5c5be4cf5dec96e7bb27'),
 'generation/src/gencuda/backends/hy_motion_motionrft.py':(63742,'c98c52e8b24dd73862beb9e9231f5008ee83698a8f3e477836cd1d0da2d8e603'),
}


def main():
    start=time.monotonic()
    out=G/'local-only/training-hymotion-local-input-provenance-v1-original'
    out.mkdir(exist_ok=False)
    archive=(G/'code/bw1000/training-framework-bd24ec3-v1/FRAMEWORK.tar.gz').read_bytes()
    assert len(archive)==738802 and hashlib.sha256(archive).hexdigest()=='85cffcb53affa1905b42afec358236089c5e7e094b9ac65d4b464ff39b6f449f'
    manifest=json.loads((G/'code/bw1000/training-qwen-target-unified-framework-a68952c-v1/FRAMEWORK_FILES.json').read_bytes())
    trees={};pins={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
        for name,(size,sha) in SPECS.items():
            member=tar.getmember('framework/'+name)
            assert member.isfile() and member.size==size
            raw=tar.extractfile(member).read()
            assert len(raw)==size and hashlib.sha256(raw).hexdigest()==sha
            assert manifest['files'][name]['bytes']==size and manifest['files'][name]['SHA256']==sha
            trees[name]=ast.parse(raw);pins[name]=dict(bytes=size,SHA256=sha)
    producer=trees['generation/scripts/prepare_hy_motion_training_cache.py']
    funcs={n.name:n for n in producer.body if isinstance(n,ast.FunctionDef)}
    defaults={}
    mlu_lines=[]
    for node in ast.walk(funcs['main']):
        if isinstance(node,ast.Call):
            if isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant):
                defaults[node.args[0].value]=next((k.value.value for k in node.keywords if k.arg=='default' and isinstance(k.value,ast.Constant)),None)
            if any(k.arg=='device' and isinstance(k.value,ast.Constant) and k.value.value=='mlu' for k in node.keywords):
                mlu_lines.append(node.lineno)
    assert defaults['--frames']==60 and defaults['--min-frames']==40 and len(mlu_lines)>=2
    selection=funcs['iter_selected_samples']
    # Exact original selector truncates to min(raw length, requested frames).
    selected_length=next(v for n in ast.walk(selection) if isinstance(n,ast.Dict) for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant) and k.value=='length')
    assert ast.unparse(selected_length)=='min(frames, args.frames)'
    padding=next(n for n in ast.walk(funcs['main']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='padded' for t in n.targets))
    assert ast.unparse(padding.value)=='np.zeros((args.frames, 201), dtype=np.float32)'
    assembly=next(n for n in ast.walk(funcs['main']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='assemble')
    assert [ast.unparse(a) for a in assembly.args]==['motion_135','keypoints']
    keypoints=next(n for n in ast.walk(funcs['main']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='keypoints' for t in n.targets))
    assert 'body_model' in ast.unparse(keypoints.value)
    assert '135' in ast.unparse(funcs['assemble']) and '66' in ast.unparse(funcs['assemble'])
    rft=trees['generation/src/gencuda/backends/hy_motion_motionrft.py']
    sample=next(n for n in rft.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_group')
    frame=next(n for n in ast.walk(sample) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='frames' for t in n.targets))
    assert ast.unparse(frame.value)=="int(sample['motion'].shape[0])"
    assert any(isinstance(n,ast.Tuple) and ast.unparse(n)=='(1, frames, 201)' for n in ast.walk(sample))
    # These are existing task references, not re-consumption of old terminal/data.
    task_paths=[G/'coordination/tasks'/n for n in ['training-hymotion-frame-contract-v1.json','training-hymotion-first-raw-pair-NPY-header-RO-v1.json']]
    task_pins={}
    for p in task_paths:
        raw=p.read_bytes();task_pins[p.name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    frame_task=json.loads(task_paths[0].read_bytes())
    missing_task=json.loads(task_paths[1].read_bytes())
    assert frame_task['actual_original_frames']==60 and frame_task['original_sample_count']==128
    assert missing_task['state']=='ACTUAL_BOTH_DECLARED_RAW_PATHS_MISSING_FRAMES_UNKNOWN'
    result=dict(schema='HYMOTION_REAL_LOCAL_PRODUCER_INPUT_PROVENANCE_V1',passed=True,request_id=REQUEST,
        source_provider_pins=pins,source_SHA_matches_current_common393=True,task_reference_pins=task_pins,
        producer_default_frames=60,producer_default_min_frames=40,selected_length='min(raw_pair_frames, requested_frames)',
        padded_buffer_shape=['requested_frames',201],zero_padding_not_evidence_of_real_128frames=True,
        target201_source='135 input features plus 66 body_model-recomputed keypoint features; not direct use of original new_joints values',
        hardcoded_MLU_device_keyword_lines=mlu_lines,producer_CPU_execution_supported=False,
        MotionRFT_frames_source='actual sample.motion.shape[0]',MotionRFT_latent_shape=[1,'frames',201],
        actual_original_cache_frames=60,original_cache_sample_count=128,real128frame_cache_qualified=False,
        declared_raw_pair_current_task_state=missing_task['state'],
        raw_motion_or_cache_body_read=0,old_terminal_read=0,producer_executed=False,
        HTTP=0,SSH=0,Torch=0,weights=0,GPU=0,training_updates=0,
        training_READY_increment=0,all_GO=False,wall_seconds=time.monotonic()-start,
        exact_missing_prerequisites=['A legally available real source motion pair with >=128 frames and all source pins; old declared exact pair was missing',
          'Native body-model/statistics/text-condition identities and a qualified producer platform; current producer explicitly invokes MLU',
          'New manifest/cache declaring128 frames and actual per-sample lengths plus matching A100 firststep/recovery reference; cannot relabel the60frame128sample cache'])
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['passed','request_id','producer_default_frames','actual_original_cache_frames','original_cache_sample_count','producer_CPU_execution_supported','real128frame_cache_qualified','training_READY_increment']}))


if __name__=='__main__':main()
