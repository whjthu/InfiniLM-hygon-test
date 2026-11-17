"""Guarded worker; fetch mode is never called by the local synthetic memory probe."""
import argparse
import io
import json
from pathlib import Path
import time


def memory_probe(remaining):
    from prepare_caption_dev import parse_decoded_json,select
    started=time.monotonic();stream=io.BytesIO();stream.write(b'{"images":[')
    images,annotations=120000,600000
    for i in range(images):
        if i%10000==0:remaining()
        identity=1000000+i;w,h=((640,480),(480,640),(500,500),(640,480))[i%4]
        record=dict(id=identity,file_name=f'{identity:012d}.jpg',coco_url=f'http://images.cocodataset.org/train2017/{identity:012d}.jpg',
            flickr_url=f'http://farm1.staticflickr.com/1/{90000000000+identity}_abcdef.jpg',license=4,width=w,height=h)
        if i:stream.write(b',')
        stream.write(json.dumps(record,separators=(',',':')).encode())
    stream.write(b'],"annotations":[')
    for i in range(annotations):
        if i%10000==0:remaining()
        if i:stream.write(b',')
        stream.write(json.dumps(dict(id=i,image_id=1000000+i%images,caption='synthetic memory probe only'),separators=(',',':')).encode())
    stream.write(b'],"licenses":[{"id":4,"url":"http://creativecommons.org/licenses/by/2.0/"}],"padding":"')
    target=91865115;pad=target-stream.tell()-2
    if pad<0:raise ValueError('synthetic memory stress exceeds intended raw size')
    block=b'x'*65536
    while pad:
        remaining();take=min(pad,len(block));stream.write(block[:take]);pad-=take
    stream.write(b'"}');raw=stream.getvalue()
    # Approximate two simultaneous compressed buffers held by actual decode path.
    compressed_buffers=[b'c'*18816148,b'd'*18816148]
    before=time.monotonic();payload=parse_decoded_json(raw);parse_seconds=time.monotonic()-before
    before=time.monotonic();selected,receipt=select(payload['images'],payload['licenses'],dict(ids=set(),photos=set(),image_SHA256=set()))
    select_seconds=time.monotonic()-before
    remaining()
    return dict(state='SYNTHETIC_ACTUAL_PARSER_SELECTOR_MEMORY_PROBE_COMPLETED',
        raw_bytes=len(raw),synthetic_images=images,synthetic_annotations=annotations,synthetic_selected=len(selected),
        simultaneous_compressed_buffer_bytes=sum(map(len,compressed_buffers)),parse_seconds=parse_seconds,
        selector_seconds=select_seconds,total_seconds=time.monotonic()-started,
        actual_official_annotation_consumed=False,actual_official_roster_selected=0,
        official_dataset_peak_memory_qualification=False,real_HTTP_calls=0,raw_synthetic_payload_saved=False)


def main(args):
    from fetch_https import remaining
    def left():return remaining(args.deadline,args.monotonic_end)-7
    def check_left():
        value=left()
        if value<=0:raise TimeoutError('original worker epoch exhausted')
        return value
    while not args.gate.is_file():
        check_left();time.sleep(0.02)
    gate=json.loads(args.gate.read_text(encoding='utf8'))
    if gate['memory_limit_bytes']!=2*1024**3 or gate['deadline']!=args.deadline:raise ValueError('memory gate differs')
    from job_guard import current_affinity
    affinity=current_affinity()
    if (gate['CPU_count']!=2 or gate['job_affinity_verified'] is not True
            or affinity['CPU_count']!=2 or affinity['process_mask']!=gate['worker_affinity_mask']):
        raise ValueError('Actual worker2CPU affinity differs')
    result=dict(state='WORKER_STARTED',mode=args.mode,planned_images=16,actual_metadata_selected=0,
                missing_slots=16,training_updates=0,GPU_calls=0,quality_metric_emitted=False,
                actual_worker_CPU_count=affinity['CPU_count'],actual_worker_affinity_mask=affinity['process_mask'])
    try:
        check_left()
        if args.mode=='smoke':
            result.update(state='LOCAL_2CPU_JOB_STARTUP_SMOKE_COMPLETED',real_HTTP_calls=0,
                          actual_official_data_consumed=False,large_memory_probe=False)
        elif args.mode=='probe':result.update(memory_probe(check_left))
        else:
            from prepare_caption_dev import prepare,protected64,train_member_plan
            G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
            cache=G/'local-only/training-qwen-generic-public-holdout-v1'
            binding=G/'code/bw1000/training-vlm-original-assets-v1/original-bindings.json'
            plan=train_member_plan(cache);protected=protected64(cache,binding)
            if plan['HTTP_range_bytes']!=18816148 or not len(protected['ids'])==len(protected['image_SHA256'])==64:
                raise ValueError('Original source plan or protected64 precondition differs')
            from fetch_https import fetch
            acquisition=fetch(args.directory,args.deadline,args.monotonic_end)
            check_left()
            receipt=prepare(args.directory/acquisition['member_file'],acquisition['member_SHA256'],
                cache,binding,args.directory/'selection')
            result.update(state='PUBLIC_MEMBER_ACQUIRED_METADATA_SELECTION_RECORDED',fetch=acquisition,
                selection=receipt,actual_metadata_selected=receipt['selected'],missing_slots=receipt['missing_slots'],
                evaluation_ready=False,training_GO=False)
        after=current_affinity()
        if after['CPU_count']!=2 or after['process_mask']!=affinity['process_mask']:
            raise ValueError('Worker CPU affinity changed')
        result['worker_affinity_unchanged_after_work']=True
    except BaseException as error:
        result.update(state='WORKER_FAILED',error_type=type(error).__name__)
        roster=args.directory/'selection/roster.json'
        if roster.is_file() and not roster.is_symlink() and roster.stat().st_size<=32768:
            try:
                count=len(json.loads(roster.read_bytes()))
                if 0<=count<=16:result.update(actual_metadata_selected=count,missing_slots=16-count,
                                             frozen_roster_survived_later_failure=True)
            except (ValueError,TypeError):pass
    with (args.directory/'worker-result.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ('state','mode','planned_images','actual_metadata_selected','missing_slots','quality_metric_emitted')}))
    return 1 if result['state']=='WORKER_FAILED' else 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fetch','probe','smoke'],required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--monotonic-end',type=float,required=True)
    raise SystemExit(main(p.parse_args()))
