"""Versioned exact byte-block wire contract; never decodes a tensor or chooses fallback."""
import hashlib,json,re
from common import P,need,canonical,sha
MAGIC=b'IVLBD01\n'

def integer(v):return type(v)is int

def validate_plan(value):
    need(type(value)is dict and set(value)=={'stage','schema','frames','total_wire_bytes','source_verified_members'},'PATCH_PLAN_KEYS')
    need(value['stage']=='PATCH_PLAN' and value['schema']==P['schema'],'PATCH_PLAN_VERSION')
    frames=value['frames'];need(type(frames)is list and len(frames)==7,'SEVEN_FRAME_DENOMINATOR')
    total=0
    for row,item in zip(P['files'],frames):
        need(type(item)is dict and set(item)=={'rank','name','bytes','SHA256','changed_blocks','changed_data_bytes'},'FRAME_DESCRIPTOR_KEYS')
        need(item['rank']==row['rank'] and item['name']==row['name'],'FROZEN_RANK_ORDER')
        need(integer(item['bytes']) and 12<item['bytes']<=P['frame_cap_bytes'],'COMPLETE_FRAME_CAP')
        need(type(item['SHA256'])is str and re.fullmatch('[0-9a-f]{64}',item['SHA256']),'FRAME_SHA_TYPE')
        need(integer(item['changed_blocks']) and 0<=item['changed_blocks']<=(row['bytes']+65535)//65536,'CHANGED_BLOCK_COUNT')
        need(integer(item['changed_data_bytes']) and 0<=item['changed_data_bytes']<item['bytes'],'CHANGED_DATA_BOUND')
        total+=item['bytes']
    need(integer(value['total_wire_bytes']) and total==value['total_wire_bytes']<=P['wire_cap_bytes'],'ALL_SEVEN_FRAMES_CAP')
    members=value['source_verified_members'];need(type(members)is list and len(members)==8,'EIGHT_ORIGINAL_SOURCE_FILES')
    for row,item in zip(P['source_files'],members):
        need(all(item[k]==row[k] for k in ['name','bytes','SHA256','rank']),'FROZEN_ORIGINAL_SOURCE_PINS')
        need(item['whole_SHA_qualified'] is True and item['stable_FD_path'] is True and item['source_stat'][2]==row['bytes'],'SOURCE_FD_PROOF')
    return value

def make_frame(row,records,blocks):
    header=dict(schema=P['schema'],rank=row['rank'],name=row['name'],base_bytes=P['base']['bytes'],base_SHA256=P['base']['SHA256'],target_bytes=row['bytes'],target_SHA256=row['SHA256'],block_bytes=P['block_bytes'],blocks=records)
    raw=canonical(header);need(len(raw)<=P['header_cap_bytes'],'HEADER_CAP')
    need(12+len(raw)+sum(map(len,blocks))<=P['frame_cap_bytes'],'COMPLETE_FRAME_CAP_NO_RAW_FALLBACK')
    return MAGIC+len(raw).to_bytes(4,'little')+raw+b''.join(blocks)

def parse_frame(raw,row,descriptor):
    need(len(raw)==descriptor['bytes']<=P['frame_cap_bytes'] and sha(raw)==descriptor['SHA256'],'FRAME_EXACT_WIRE_SHA')
    need(raw[:8]==MAGIC and len(raw)>=12,'FRAME_MAGIC')
    n=int.from_bytes(raw[8:12],'little');need(0<n<=P['header_cap_bytes'] and 12+n<=len(raw),'BOUNDED_HEADER_LENGTH')
    h=json.loads(raw[12:12+n]);need(canonical(h)==raw[12:12+n],'CANONICAL_FRAME_HEADER')
    need(set(h)=={'schema','rank','name','base_bytes','base_SHA256','target_bytes','target_SHA256','block_bytes','blocks'},'EXACT_HEADER_KEYS')
    need(h['schema']==P['schema'] and h['rank']==row['rank'] and h['name']==row['name'] and h['target_bytes']==row['bytes'] and h['target_SHA256']==row['SHA256'],'EXACT_ORIGINAL_RANK_BINDING')
    need(h['base_bytes']==P['base']['bytes'] and h['base_SHA256']==P['base']['SHA256'] and h['block_bytes']==65536,'EXACT_BASE_BLOCK_BINDING')
    need(type(h['blocks'])is list and len(h['blocks'])==descriptor['changed_blocks'],'EXACT_BLOCK_DENOMINATOR')
    offset=12+n;last=-1;result=[];data_bytes=0
    for rec in h['blocks']:
        need(type(rec)is dict and set(rec)=={'index','bytes','SHA256'} and integer(rec['index']) and rec['index']>last,'STRICT_UNIQUE_INCREASING_INDICES')
        start=rec['index']*65536;need(start<row['bytes'],'BLOCK_START_BOUND')
        size=min(65536,row['bytes']-start);need(integer(rec['bytes']) and rec['bytes']==size and offset+size<=len(raw),'EXACT_FULL_OR_LAST_BLOCK_LENGTH')
        block=raw[offset:offset+size];need(sha(block)==rec['SHA256'],'CHANGED_BLOCK_SHA')
        result.append((start,block));last=rec['index'];offset+=size;data_bytes+=size
    need(offset==len(raw) and data_bytes==descriptor['changed_data_bytes'],'NO_TRAILING_OR_MISSING_BLOCK_BYTES')
    return result
