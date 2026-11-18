"""Read one already-pinned provenance JSON; no NPY or cache tensor reads."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import time


def main(binding,context):
    assert binding['root_GO'] is True and binding['RO_GO'] is True
    deadline=context['epoch']+60;assert 0<=time.time()-context['epoch']<50
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    allowed=sorted(os.sched_getaffinity(0)-{0,1});assert len(allowed)>=2;os.sched_setaffinity(0,allowed[:2])
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    H=lambda b:hashlib.sha256(b).hexdigest();identity=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    root=Path(binding['cache_root']);pins={};total=0
    def read(name,expected,cap):
        nonlocal total
        assert name=='dataset-manifest.json'
        p=root/name;s=p.lstat();assert stat.S_ISREG(s.st_mode) and 0<s.st_size<=cap
        with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb',buffering=0) as f:
            before=os.fstat(f.fileno());raw=f.read(s.st_size);after=os.fstat(f.fileno())
        total+=len(raw);assert total<=1024**2
        assert len(raw)==s.st_size and identity(s)==identity(before)==identity(after)==identity(p.stat()) and H(raw)==expected
        pins[name]=dict(bytes=len(raw),SHA256=H(raw));return json.loads(raw)
    dataset=read('dataset-manifest.json',binding['dataset_manifest_SHA256'],1024**2)
    assert dataset['frames']==60 and dataset['feature_dim']==201 and dataset['sample_count']==128
    assert len(dataset['samples'])==128
    rows=[];seen=set()
    for sample in dataset['samples']:
        assert sample['sample_id'] not in seen;seen.add(sample['sample_id'])
        assert type(sample['length'])is int and 1<=sample['length']<=60
        sources=sample['sources'];assert set(sources)=={'rot6d','joints','text'}
        # These are identities from the pinned source manifest, not files followed/read here.
        raw_fields={key:sample[key] for key in ('source_frames','raw_frames','raw_length','frames_before_crop') if key in sample}
        rows.append(dict(sample_id_SHA256=H(sample['sample_id'].encode()),cached_valid_length=sample['length'],sources=sources,
                         caption_SHA256=H(sample['prompt'].encode()) if isinstance(sample.get('prompt'),str) else None,
                         declared_raw_length_fields=raw_fields,source_frames=raw_fields.get('source_frames')))
    roots=set()
    for row in rows:
        rot=Path(row['sources']['rot6d']['path']);joints=Path(row['sources']['joints']['path']);text=Path(row['sources']['text']['path'])
        if rot.is_absolute() and rot.parent.name=='joints_6d' and joints.parent.name=='new_joints' and text.parent.name=='texts' and rot.parent.parent==joints.parent.parent==text.parent.parent:
            roots.add(str(rot.parent.parent))
    result=dict(context=context,metadata_pins=pins,explicit_metadata_bytes=total,source_rows=rows,
                source_repository=dataset.get('source_repository'),source_archive=dataset.get('source_archive'),
                source_archive_SHA256=dataset.get('source_archive_sha256'),model_statistics=dataset.get('model_statistics'),
                declared_raw_dataset_roots=sorted(roots),dataset_frames=dataset['frames'],feature_dim=dataset['feature_dim'],
                cached_length_min=min(s['cached_valid_length'] for s in rows),cached_length_max=max(s['cached_valid_length'] for s in rows),
                cached_short_valid_length_count=sum(s['cached_valid_length']<dataset['frames'] for s in rows),
                raw_length_field_row_count=sum(bool(s['declared_raw_length_fields']) for s in rows),
                actual_padding_tensor_values_verified=False,
                true_raw_length_ge128='UNKNOWN_UNTIL_EXACT_NPY_HEADERS',cached_tensor_or_raw_motion_bytes_read=0,
                captions_returned=False,np_Torch_or_producer_imported=False,GPU_or_Slurm=False)
    if rows:
        result['next_single_header_pair_only']=dict(sample_id_SHA256=rows[0]['sample_id_SHA256'],
            sources={k:rows[0]['sources'][k] for k in ('rot6d','joints')},header_max_bytes_per_file=4096,
            array_body_read=False,selection='first original manifest row, not by observed quality/length; future separately approved scope')
    raw=json.dumps(result,separators=(',',':'),allow_nan=False).encode();assert len(raw)<=131072
    os.write(1,raw+b'\n')


if __name__=='__main__':main(BINDING,CONTEXT)
