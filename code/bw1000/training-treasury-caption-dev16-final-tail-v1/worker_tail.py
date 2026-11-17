"""One2CPU/2GiB tail worker, fresh600s epoch; original failed scope remains immutable."""
import argparse
import json
from pathlib import Path
import time


def main(a):
    from tail_https import left
    def remaining():return left(a.deadline,a.monotonic_end)
    while not a.gate.is_file():remaining();time.sleep(0.02)
    from job_guard import current_affinity
    gate=json.loads(a.gate.read_text(encoding='utf8'));affinity=current_affinity()
    if (gate['deadline']!=a.deadline or gate['memory_limit_bytes']!=2147483648 or gate['CPU_count']!=2
            or gate['job_affinity_verified'] is not True or affinity['CPU_count']!=2
            or affinity['process_mask']!=gate['worker_affinity_mask']):raise ValueError('Actualtail Job2CPU/2GiB gate differs')
    result=dict(state='TAIL_WORKER_STARTED',planned_images=16,actual_metadata_selected=0,missing_slots=16,
        actual_worker_CPU_count=2,actual_worker_affinity_mask=affinity['process_mask'],
        original_scope_still_failed=True,old_epoch_not_extended=True,HTTP_prefix_bytes_requested=0,
        quality_metric_emitted=False,evaluation_ready=False,training_GO=False,GPU_GO=False)
    try:
        from chain import G,consumer_record,copy_prefix
        original=consumer_record()
        from prepare_caption_dev import protected64,train_member_plan,prepare
        cache=G/'local-only/training-qwen-generic-public-holdout-v1'
        binding=G/'code/bw1000/training-vlm-original-assets-v1/original-bindings.json'
        protected=protected64(cache,binding);plan=train_member_plan(cache)
        if not len(protected['ids'])==len(protected['image_SHA256'])==64 or plan['HTTP_range_bytes']!=18816148:
            raise ValueError('Originalcaption source plan/protection differs')
        assembled=a.directory/'captions_train2017.assembled.partial'
        digest,prefix=copy_prefix(assembled,original,remaining)
        result['local_prefix_copy']=prefix
        from tail_https import acquire_tail
        tail=acquire_tail(a.directory,assembled,digest,a.deadline,a.monotonic_end)
        result['tail']=tail;remaining()
        # Published original decoder enforces original complete ZIP header/size/CRC,
        # exact91865115B decoded bytes and SHA before metadata-only fixed-seed selection.
        selection=prepare(a.directory/tail['member_file'],tail['assembled_SHA256'],cache,binding,a.directory/'selection')
        result.update(state='SUFFIX_ACQUIRED_COMPLETE_MEMBER_CRC_AND_METADATA_SELECTION_RECORDED',
            selection=selection,actual_metadata_selected=selection['selected'],missing_slots=selection['missing_slots'])
        after=current_affinity()
        if after['CPU_count']!=2 or after['process_mask']!=affinity['process_mask']:raise ValueError('Tail worker affinity changed')
        result['worker_affinity_unchanged_after_work']=True
    except BaseException as error:
        result.update(state='TAIL_WORKER_FAILED',error_type=type(error).__name__)
        roster=a.directory/'selection/roster.json'
        if roster.is_file() and not roster.is_symlink() and roster.stat().st_size<=32768:
            try:
                count=len(json.loads(roster.read_bytes()))
                if 0<=count<=16:result.update(actual_metadata_selected=count,missing_slots=16-count,frozen_roster_survived_later_failure=True)
            except (ValueError,TypeError):pass
    with (a.directory/'worker-result.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('state','planned_images','actual_metadata_selected','missing_slots','HTTP_prefix_bytes_requested','quality_metric_emitted')}))
    return 1 if result['state']=='TAIL_WORKER_FAILED' else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--monotonic-end',type=float,required=True)
    raise SystemExit(main(p.parse_args()))
