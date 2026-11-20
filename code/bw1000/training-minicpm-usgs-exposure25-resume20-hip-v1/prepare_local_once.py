"""Build actual MiniCPM exposure view and both common-framework command plans."""
import ast
import hashlib
import json
from pathlib import Path
import time

from data_view import build
from minicpm_domain_recipe import command_plan,require_native88

D=Path(__file__).resolve().parent
G=D.parents[2]
REQUEST='87bb4872406c4f4fa48de3adc9e0b2ad'
RUN=G/'local-only/training-minicpm-usgs-exposure25-resume20-hip-v1-preparation'
TARGET='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-usgs-exposure25-resume20-hip-v1'


def pin(raw):return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def main():
    origin=time.time()
    RUN.mkdir(exist_ok=False)
    original=G/'local-only/training-domain-usgs-catalog-table-reading-v1-qwen-view'
    source=original/'train/swift.jsonl'
    assets=json.loads((D/'ASSETS.json').read_bytes())
    metadata=source.read_bytes()
    assert pin(metadata)==dict(bytes=29658,SHA256=assets['dataset_SHA256'])
    records=[json.loads(line) for line in metadata.splitlines()]
    assert len(records)==88 and len({r['id'] for r in records})==88
    media_pins={};image_paths=set()
    for row in records:
        assert [m['role'] for m in row['messages']]==['user','assistant'] and len(row['images'])==1
        image=row['images'][0]
        prefix='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-domain-usgs-table-reading-v1/data/train/media/'
        assert image.startswith(prefix)
        name=image[len(prefix):]
        assert Path(name).name==name and name in assets['media']
        image_paths.add(image)
        if name not in media_pins:
            path=original/'train/media'/name
            assert path.is_file() and not path.is_symlink()
            raw=path.read_bytes();assert pin(raw)==assets['media'][name]
            media_pins[name]=pin(raw)
    assert len(media_pins)==len(image_paths)==22
    view=build(source,RUN/'view')
    raw=(RUN/'view/training800.jsonl').read_bytes()
    lines=raw.splitlines(keepends=True);original_lines=metadata.splitlines(keepends=True)
    assert len(lines)==800 and all(line==original_lines[i%88] for i,line in enumerate(lines))
    translator=G/'code/bw1000/training-qwen-target-native-CPU-v1/deltas/tools/hygon_vlm_recipe.py'
    plans={}
    for arm in ['continuous','resumed']:
        plan=command_plan(translator,(D/'original-launch.json').read_bytes(),TARGET,view['view']['SHA256'],arm,29929)
        body=json.dumps(plan,sort_keys=True,indent=2).encode()+b'\n'
        (RUN/(arm+'-plan-private.json')).write_bytes(body)
        plans[arm]=pin(body)
    # The source contract is an actual gate: no CPU88 result exists/pinned yet.
    native_missing_rejected=False
    try:require_native88(b'{}',None)
    except ValueError:native_missing_rejected=True
    assert native_missing_rejected
    for path in D.glob('*.py'):ast.parse(path.read_bytes(),str(path))
    result=dict(schema='MINICPM_USGS88_EXPOSURE800_ACTUAL_LOCAL_PREPARATION_V1',passed=True,
        request_id=REQUEST,planned_source_rows=88,actual_source_rows=88,actual_train_media_SHA_verified=22,
        source_train_file=pin(metadata),view=view['view'],physical_exposures=800,new_independent_examples=0,
        min_exposures=9,max_exposures=10,original_lines_retained=True,
        planned_world_size=8,planned_micro_per_rank=100,accumulation=4,
        planned_continuous_updates=25,planned_resume20_updates=5,planned_total_native_updates=30,
        checkpoint20_native_skips_per_rank=80,resume_tail_micro_per_rank=20,
        expected_native_checkpoint_members=14,expected_checkpoint_steps=[10,20,25],
        original_initial_named_SHA256=assets['initial']['named_SHA256'],
        exact_common_T05_entry_plan_count=2,plans=plans,native88_missing_rejected=True,
        shared_device_training_loop='existing Swift Trainer via frozen common T05 build_launch',
        fresh_original_trainables_restored_by_callback=True,
        original_warmup5_LR0_step_allowed=True,optimizer_counters_must_advance=True,
        all25_no_parameter_change_rejected=True,
        CPU_native88_actual_qualified=False,target_installed=False,target_READY=False,
        actual_encode_calls=0,actual_optimizer_updates=0,HTTP=0,SSH=0,Torch=0,GPU=0,weights_body_read=0,
        dev_or_blind_input_gold_media_read=False,
        prior_CPU_plan_failure='first local attempt expected original callback in resumed plan; common builder already removes it. Corrected to exact expected0; no child launched.',
        wall_seconds=time.time()-origin,training_READY_increment=0)
    (RUN/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ['passed','request_id','actual_source_rows','actual_train_media_SHA_verified','physical_exposures','exact_common_T05_entry_plan_count','native88_missing_rejected','actual_optimizer_updates','target_READY']}))


if __name__=='__main__':main()
