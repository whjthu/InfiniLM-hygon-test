"""Freeze only16 existing media identities and user prompts, never open images/references."""
import argparse
import hashlib
import json
from pathlib import Path

G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
MEDIA_ROOT=G/'local-only/training-treasury-caption-dev16-media-stage-v1-original/media'
SELECTION=G/'local-only/training-treasury-caption-dev16-final-tail-v1-original/selection'
AUDIT=G/'local-only/training-treasury-caption-dev16-v1/media-original-independent-result.json'
PROMPT='<image>\nDescribe this image in one short sentence.'


def pinned(path,digest):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Original small metadata SHA differs')
    return json.loads(raw)


def prepare(output):
    if output.exists() or not output.resolve().is_relative_to((G/'local-only').resolve()):raise ValueError('New exclusive local-only preparation directory required')
    roster=pinned(SELECTION/'roster.json','96eaeb4b4bd5f52bcf6fbd39370f61fc1ed24ea22842dc77e15c69a5c1885627')
    inputs=pinned(SELECTION/'inputs.json','a2f97e20027a9223d43828e92abba1baf82e5b1122f195c5eca558965d50212c')
    audit=pinned(AUDIT,'9becb5ef8a50f5545af1f4bd387e06bf46d5b6415bf7401e0b3aa570f195c94f')
    if not(len(roster)==len(inputs)==len(audit['private_ordered_slot_evidence'])==16 and audit['passed'] and audit['independently_qualified_media_images']==16):raise ValueError('Original16 audit denominator differs')
    rows=[];images=[]
    for i,(row,request,evidence) in enumerate(zip(roster,inputs,audit['private_ordered_slot_evidence'])):
        name=row['file_name'];pin=evidence['actual_file']
        if not(request==dict(id='coco:'+str(row['coco_image_id']),messages=[dict(role='user',content=PROMPT)],images=['media/'+name])
               and name==f"{row['coco_image_id']:012d}.jpg" and evidence['slot']==i and evidence['qualified'] is True
               and pin['passive_header_passed'] is True and (pin['actual_width'],pin['actual_height'])==(row['width'],row['height'])):raise ValueError('Original prompt/media slot identity differs')
        rows.append(request)
        images.append(dict(slot=i,name='media/'+name,source=(MEDIA_ROOT/name).as_posix(),bytes=pin['bytes'],SHA256=pin['SHA256'],width=row['width'],height=row['height']))
    output.mkdir();raw=(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows)).encode('utf8')
    (output/'inputs.jsonl').write_bytes(raw)
    manifest=dict(state='FIXED16_ANSWER_FREE_INPUTS_FROZEN_NOT_STAGED',planned_rows=16,images=images,
        inputs=dict(name='inputs.jsonl',bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()),
        original_inputs_SHA256='a2f97e20027a9223d43828e92abba1baf82e5b1122f195c5eca558965d50212c',
        roster_SHA256='96eaeb4b4bd5f52bcf6fbd39370f61fc1ed24ea22842dc77e15c69a5c1885627',
        media_independent_result_SHA256='9becb5ef8a50f5545af1f4bd387e06bf46d5b6415bf7401e0b3aa570f195c94f',
        image_bytes=sum(p['bytes'] for p in images),images_read_here=0,references_read=0,labels_in_inputs=False,
        author_licensing_pending=True,Dataset_GO=False,training_GO=False,GPU_GO=False)
    body=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode('utf8');(output/'DATA_MANIFEST.json').write_bytes(body)
    print(json.dumps(dict(planned_rows=16,image_bytes=manifest['image_bytes'],inputs_SHA256=manifest['inputs']['SHA256'],DATA_MANIFEST_SHA256=hashlib.sha256(body).hexdigest(),real_images_read=0,references_read=0,actual_native_processor_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
