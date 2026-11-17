"""Metadata-only fixed16 COCO train2017 development roster; no network/model/scorer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
from urllib.parse import urlsplit
import zlib

ORIGINAL_BINDING_SHA='5d9e6132e937da172dd714313ba1dfdc7775e86b27fb623131fe9bbe2c1d1822'
TAIL_SHA='a7276b808dd35e1aaff18ca909c0854d303361a7b550e75e7e2689341a30dbe4'
VAL_BODY_SHA='ba2348b66158aea7083493f3152b8a6c8f8fc5ee1e46df0a48741526941ce7f3'
VAL_HEADER_SHA='80e5e7f1592d23905a4d51ed6e179177e6120c303d6747840b368f9b2fede6fd'
VAL_JSON_SHA='afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9'
MEMBER=b'annotations/captions_train2017.json'
SEED=hashlib.sha256(b'Treasury-generic-caption-dev16-v1-20261117').hexdigest()
QUOTAS={'landscape':8,'portrait':4,'square':4}
COUNT=16
PROMPT='<image>\nDescribe this image in one short sentence.'
FLICKR_HOST=re.compile(r'(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com|farm[0-9]+\.static\.flickr\.com')
FLICKR_PATH=re.compile(r'/(?:[0-9]+/){1,3}([0-9]+)_[0-9a-f]+(?:_[a-z0-9]+)?\.(?:jpg|jpeg|png|gif)',re.I)


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(data):return hashlib.sha256(data).hexdigest()


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()


def pinned(path,digest,limit):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size<=limit,'unsafe local metadata')
    data=path.read_bytes();require(sha(data)==digest,'metadata SHA differs');return data


def photo(url):
    part=urlsplit(url or '');match=FLICKR_PATH.fullmatch(part.path)
    return match[1] if part.scheme in ('http','https') and FLICKR_HOST.fullmatch(part.hostname or '') and match else None


def train_member_plan(cache):
    tail=pinned(cache/'coco-annotation-zip-tail.bin',TAIL_SHA,65536)
    base=252842005;end=tail.rfind(b'PK\x05\x06')
    require(end>=0,'ZIP end directory absent')
    _,disk,cd_disk,count,total,cd_size,cd_start,comment=struct.unpack_from('<4s4H2IH',tail,end)
    require(disk==cd_disk==0 and count==total==6 and base+len(tail)==252907541
            and end+22+comment==len(tail),'ZIP central directory identity differs')
    cursor=cd_start-base;entries=[]
    for _ in range(count):
        fields=struct.unpack_from('<4s6H3I5H2I',tail,cursor);n,e,c=fields[10:13]
        require(fields[0]==b'PK\x01\x02' and fields[13]==0,'ZIP directory entry differs')
        name=tail[cursor+46:cursor+46+n]
        entries.append(dict(name=name.decode(),offset=fields[16],flags=fields[3],method=fields[4],
            crc32=fields[7],compressed_bytes=fields[8],uncompressed_bytes=fields[9],name_bytes=n))
        cursor+=46+n+e+c
    require(cursor==cd_start-base+cd_size,'ZIP directory end differs')
    index=next(i for i,e in enumerate(entries) if e['name']==MEMBER.decode())
    item=entries[index];following=entries[index+1]['offset']
    require((item['offset'],item['flags'],item['method'],item['crc32'],item['compressed_bytes'],item['uncompressed_bytes'])==
            (159287046,0,8,3427477015,18816055,91865115),'train caption member differs')
    return dict(**item,range_start=item['offset'],range_end=following-1,HTTP_range_bytes=following-item['offset'],
        local_extra_length_inferred=following-item['offset']-30-item['name_bytes']-item['compressed_bytes'],
        archive_total_bytes=252907541,central_directory_SHA256=TAIL_SHA,
        official_URL='http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        future_response_requirement='Exactly206 with expected Content-Range/length; refuse200/no fallback; no full archive.',
        dataset_HTTP_bytes_executed=0,fetch_authorized_by_this_program=False)


def protected64(cache,binding):
    original=json.loads(pinned(binding,ORIGINAL_BINDING_SHA,1024**2))['common']['image_identity']
    h=pinned(cache/'coco-caption-val-member-header.bin',VAL_HEADER_SHA,30)
    body=pinned(cache/'coco-caption-val-member.bin',VAL_BODY_SHA,805766)
    fields=struct.unpack('<4s5H3I2H',h);n,e=fields[-2:]
    require(body[:n]==b'annotations/captions_val2017.json','original val member differs')
    compressed=body[n+e:];decoded=zlib.decompress(compressed,-15)
    require(len(decoded)==3872473 and zlib.crc32(decoded)==1836330781 and sha(decoded)==VAL_JSON_SHA,
            'original val publisher identity differs')
    # Selection/protection uses image metadata only; no caption text is accessed.
    images={row['id']:row for row in json.loads(decoded)['images']}
    ids=set();hashes=set();photos=set()
    require(len(original)==64 and len(images)==5000,'original image denominator differs')
    for index,row in enumerate(original):
        role=row['relative_role'];match=re.fullmatch(r'coco-val2017/images/([0-9]{12})\.jpg',role)
        require(match and row['ordered_index']==index and re.fullmatch('[0-9a-f]{64}',row['sha256']),
                'original64 binding schema differs')
        identity=int(match[1]);require(identity in images,'original64 publisher membership absent')
        flickr=photo(images[identity]['flickr_url']);require(flickr is not None,'original Flickr group absent')
        ids.add(identity);hashes.add(row['sha256']);photos.add(flickr)
    require(len(ids)==len(hashes)==64,'original64 ID/content duplicates')
    return dict(ids=ids,image_SHA256=hashes,photos=photos)


def select(images,licenses,protected):
    """No annotation/caption/answer/model-output argument is accepted here."""
    mapping={row['id']:row for row in licenses}
    require(mapping.get(4,{}).get('url') in ('http://creativecommons.org/licenses/by/2.0/',
            'https://creativecommons.org/licenses/by/2.0/'),'publisher image license4 differs')
    ids=[row.get('id') for row in images]
    require(all(type(i) is int and i>0 for i in ids) and len(ids)==len(set(ids)),'candidate image IDs invalid/repeated')
    rejected=Counter();admitted=[]
    for row in images:
        identity=row['id'];name=f'{identity:012d}.jpg';url=urlsplit(row.get('coco_url',''))
        flickr=photo(row.get('flickr_url'))
        if row.get('file_name')!=name or url.hostname!='images.cocodataset.org' or url.path!='/train2017/'+name:
            rejected['wrong_train_membership_or_URL']+=1;continue
        if identity in protected['ids']:
            rejected['original64_COCO_ID_overlap']+=1;continue
        if flickr is None:
            rejected['source_group_unknown']+=1;continue
        if flickr in protected['photos']:
            rejected['original64_Flickr_group_overlap']+=1;continue
        if row.get('license')!=4:
            rejected['outside_frozen_CC_BY2_image_policy']+=1;continue
        width,height=row.get('width'),row.get('height')
        if type(width) is not int or type(height) is not int or min(width,height)<=0:
            rejected['invalid_dimensions']+=1;continue
        group='landscape' if width>height else 'portrait' if width<height else 'square'
        admitted.append((sha(f'{SEED}:{identity:012d}'.encode()),row,flickr,group))
    selected=[];used=set();counts=Counter()
    for _,row,flickr,group in sorted(admitted,key=lambda item:(item[0],item[1]['id'])):
        if flickr in used:rejected['candidate_source_group_duplicate']+=1;continue
        used.add(flickr)
        if counts[group]>=QUOTAS[group]:rejected['eligible_outside_frozen_quota']+=1;continue
        selected.append(dict(coco_image_id=row['id'],file_name=row['file_name'],coco_url=row['coco_url'],
            flickr_static_url=row['flickr_url'],flickr_photo_id=flickr,license_id=4,
            license_url='https://creativecommons.org/licenses/by/2.0/',width=row['width'],height=row['height'],stratum=group))
        counts[group]+=1
    return selected,dict(pool_images=len(images),selected=len(selected),planned=COUNT,missing_slots=COUNT-len(selected),
        quotas=QUOTAS,selected_by_stratum=dict(counts),rejected=dict(rejected),
        selection_seed_SHA256=SEED,selection_uses_captions_or_model_outputs=False,
        original64_ID_and_Flickr_overlap=0,actual_candidate_content_SHA_overlap='NOT_MEASURED',
        near_duplicate_content_overlap='NOT_MEASURED',scorer_qualified=False,formal_or_blind_CIDEr=False)


def decoded_train(record,expected_sha):
    require(re.fullmatch('[0-9a-f]{64}',expected_sha),'actual acquired member SHA required')
    data=pinned(record,expected_sha,18816148)
    require(len(data)==18816148,'partial train member not allowed')
    fields=struct.unpack_from('<4s5H3I2H',data);signature,_,flags,method,_,_,crc,n,size,name_n,extra_n=fields
    require(signature==b'PK\x03\x04' and (flags,method,crc,n,size)==(0,8,3427477015,18816055,91865115)
            and data[30:30+name_n]==MEMBER and name_n==35 and extra_n==28,
            'original complete train member header differs')
    payload=data[30+name_n+extra_n:];decoder=zlib.decompressobj(-15)
    raw=decoder.decompress(payload,size+1)
    require(len(payload)==n and len(raw)==size and decoder.eof and not decoder.unconsumed_tail
            and not decoder.unused_data and zlib.crc32(raw)==crc,'train member CRC/completeness differs')
    return parse_decoded_json(raw),sha(raw)


def parse_decoded_json(raw):
    require(len(raw)<=91865115,'decoded JSON exceeds original member bound')
    return json.loads(raw)


def private_output(path):
    resolved=path.resolve();text=resolved.as_posix()
    local='C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test/local-only/'
    remote='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/'
    require(text.startswith(local) or text.startswith(remote),'raw references must remain ignored local-only or R/results')
    require(not path.exists(),'existing output must not be overwritten')
    path.mkdir()


def private_artifact_bytes(name,value):
    raw=canonical(value)
    caps={'roster.json':32768,'references.json':262144,'inputs.json':32768,'receipt.json':32768}
    require(name in caps and len(raw)<=caps[name],'selected private artifact exceeds small-output bound')
    return raw


def prepare(record,expected_sha,cache,binding,output):
    protected=protected64(cache,binding);payload,annotation_sha=decoded_train(record,expected_sha)
    # Freeze roster on image metadata before consulting any selected reference values.
    selected,receipt=select(payload['images'],payload['licenses'],protected)
    private_output(output)
    def write(name,value):
        raw=private_artifact_bytes(name,value)
        with (output/name).open('xb') as stream:stream.write(raw)
        return dict(bytes=len(raw),SHA256=sha(raw))
    roster_pin=write('roster.json',selected)
    selected_ids={row['coco_image_id'] for row in selected}
    refs={identity:[] for identity in selected_ids};ann_ids=set()
    for row in payload['annotations']:
        require(row['id'] not in ann_ids,'caption annotation IDs repeated');ann_ids.add(row['id'])
        if row['image_id'] in selected_ids:
            require(isinstance(row['caption'],str) and row['caption'].strip()
                    and len(row['caption'])<=4096,'selected reference absent or oversized')
            refs[row['image_id']].append(dict(annotation_id=row['id'],caption=row['caption']))
            require(len(refs[row['image_id']])<=10,'selected reference count exceeds fixed bound')
    references_missing=sum(not values for values in refs.values())
    ref_pin=write('references.json',refs)
    requests=[dict(id='coco:'+str(row['coco_image_id']),messages=[dict(role='user',content=PROMPT)],
                   images=['media/'+row['file_name']]) for row in selected]
    request_pin=write('inputs.json',requests)
    receipt.update(state='METADATA_ROSTER_FROZEN_MEDIA_RIGHTS_NATIVE_AND_SCORER_PENDING',
        annotation_record_SHA256=expected_sha,annotation_decoded_SHA256=annotation_sha,
        original64_binding_SHA256=ORIGINAL_BINDING_SHA,roster=roster_pin,references=ref_pin,inputs=request_pin,
        missing_reference_groups=references_missing,reference_counts=[len(refs[row['coco_image_id']]) for row in selected],
        reference_payload_private=True,caption_text_used_to_choose_roster=False,images_downloaded=0,
        image_content_nonoverlap_with_original64='PENDING_ACTUAL_STAGING',
        photographer_attribution_receipts='MISSING',native_processor='NOT_RUN',
        CIDEr_status='NOT_RUN_ACTUAL_FROZEN_SCORER_MISSING',quality_metric_emitted=False,evaluation_ready=False,
        no_backfill_after_media_or_scoring_failure=True,
        model_arms=['base','old_domain','explicit328'],training_updates=0,formal17_cases_changed=False)
    write('receipt.json',receipt)
    return {key:value for key,value in receipt.items() if key not in ('reference_counts',)}


def require_new_sha(value,protected):
    require(re.fullmatch('[0-9a-f]{64}',value) and value not in protected['image_SHA256'],
            'original64 image SHA reused or invalid')


def image_hash_gate(path,protected):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2,'unsafe or oversized candidate JPEG')
    data=path.read_bytes();require(data[:2]==b'\xff\xd8','candidate not JPEG')
    value=sha(data);require_new_sha(value,protected)
    return dict(bytes=len(data),SHA256=value)


def media_gate(roster,media_root,protected):
    require(len(roster)==COUNT and len({r['coco_image_id'] for r in roster})==COUNT
            and len({r['flickr_photo_id'] for r in roster})==COUNT,'frozen16 roster differs')
    hashes=set();checked=[];failures=Counter()
    for row in roster:
        name=row['file_name']
        require(name==f"{row['coco_image_id']:012d}.jpg",'unsafe roster image name')
        try:
            identity=image_hash_gate(media_root/name,protected)
            require(identity['SHA256'] not in hashes,'candidate16 content repeated')
            hashes.add(identity['SHA256']);checked.append(identity)
        except (OSError,ValueError):failures['missing_or_rejected_actual_media']+=1
    return dict(planned_images=COUNT,actual_checked_images=len(checked),missing_or_rejected=COUNT-len(checked),
        failures=dict(failures),exact_content_nonoverlap_under_original64_pins=len(checked)==COUNT,
        actual_media_pins=checked,near_duplicates='NOT_MEASURED',native_processor='NOT_RUN',
        photographer_attribution_receipts='MISSING',CIDEr_status='NOT_RUN_ACTUAL_FROZEN_SCORER_MISSING',
        quality_metric_emitted=False,evaluation_ready=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--original-bindings',type=Path,required=True)
    p.add_argument('--train-member-record',type=Path,required=True);p.add_argument('--train-member-sha256',required=True)
    p.add_argument('--private-output',type=Path,required=True);a=p.parse_args()
    result=prepare(a.train_member_record,a.train_member_sha256,a.cache,a.original_bindings,a.private_output)
    print(json.dumps(result,sort_keys=True))
