"""Audit derived train/dev images and metadata. Never opens QA or blind bodies."""
import hashlib
import io
import json
from pathlib import Path


def pin(raw):
    return dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())


def main():
    from PIL import Image
    G = Path(__file__).resolve().parents[3]
    run = G/'local-only/training-sdxl-USGS-style18-local-v1-original'
    data = run/'dataset'
    outcome = json.loads((run/'outcome.json').read_bytes())
    assert outcome['passed'] is True and outcome['Job']['returncode'] == 0
    assert outcome['Job']['WindowsJobObject_limits_verified'] and outcome['Job']['worker_CPU_count'] == 2
    manifest_path = G/'local-only/training-USGS-multievent-render144-v2-original/dataset/manifest-private.json'
    raw = manifest_path.read_bytes()
    assert pin(raw) == dict(bytes=14864, SHA256='4fbf6f2f8c03731eb84882fd56f63f2c4586f92d5c5d149b9dfd667973f961dc')
    original = json.loads(raw)
    binding_path = data/'media-binding-private.json'
    raw = binding_path.read_bytes()
    binding_pin = pin(raw)
    binding = json.loads(raw)
    assert len(binding['images']) == 18
    source = {r['image']:r for r in original['charts']}
    groups = {s:{r['group'] for r in original['charts'] if r['split']==s} for s in ('train','dev','blind')}
    events = {s:{e for r in original['charts'] if r['split']==s for e in r['event_ids']} for s in groups}
    for a,b in [('train','dev'),('train','blind'),('dev','blind')]:
        assert not groups[a]&groups[b] and not events[a]&events[b]
    counts, seen, pixels, total = dict(train=0,dev=0), set(), set(), 0
    for record in binding['images']:
        rel = record['transformed_relative']
        parts = Path(rel).parts
        assert len(parts)==2 and parts[0] in counts
        chart = source[record['source_relative']]
        assert chart['split']==parts[0] and chart['group']==record['group'] and chart['event_ids']==record['event_ids']
        assert record['source']==original['outputs'][record['source_relative']]
        raw = (data/rel).read_bytes()
        assert pin(raw)==record['transformed'] and pin(raw)['SHA256'] not in seen
        seen.add(pin(raw)['SHA256'])
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            assert image.format=='PNG' and image.mode=='RGB' and image.size==(256,256)
            rgb = image.tobytes()
            assert len(rgb)==196608 and min(rgb)<max(rgb)
            h=hashlib.sha256(rgb).hexdigest()
            assert h==record['RGB_pixels_SHA256'] and h not in pixels
            pixels.add(h)
        counts[parts[0]]+=1
        total+=len(raw)
    assert counts==dict(train=12,dev=6)
    captions=set()
    for split in counts:
        raw=(data/split/'metadata.jsonl').read_bytes()
        assert pin(raw)==binding['metadata'][split+'/metadata.jsonl']
        rows=[json.loads(line) for line in raw.splitlines()]
        assert len(rows)==counts[split]
        assert len({row['file_name'] for row in rows})==len(rows)
        for row in rows:
            assert set(row)=={'file_name','text'} and Path(row['file_name']).name==row['file_name']
            record=next(r for r in binding['images'] if r['transformed_relative']==split+'/'+row['file_name'])
            assert hashlib.sha256(row['text'].encode()).hexdigest()==record['caption_SHA256']
            captions.add(row['text'])
    result=dict(schema='SDXL_STYLE18_INDEPENDENT_LOCAL_AUDIT_V1',passed=True,
                planned_images=18,complete_decoded_images=18,split_images=counts,
                source_groups={s:len(v) for s,v in groups.items()}, source_events={s:len(v) for s,v in events.items()},
                cross_split_group_or_event_overlap=0,body_duplicates=0,transformed_pixel_duplicates=0,
                unique_captions=len(captions),transformed_PNG_bytes=total,media_binding=binding_pin,
                blind_media_or_gold_read=False,QA_gold_read=False,HTTP=0,Torch=0,GPU=0,optimizer_updates=0,
                target_READY=False,native_training_dataset_interface_qualified=False,
                image_text_training_view_prepared=True,semantic_numeric_generation_qualified=False)
    out=run/'audit.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
