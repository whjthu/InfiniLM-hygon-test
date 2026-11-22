"""Pure ordered train-only transformation; no data I/O or synthetic media."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import PurePosixPath

R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
DATA=R+'/results/training-gemma-usgs-render72-native-CPU-v1/data'
TRAIN=DATA+'/train-swift.jsonl'
TRAIN_PIN=dict(bytes=54246,SHA256='a897e7d9e7823d01bd2b503104d138f4a50975cabc06a2fefcda1dfb36ff05fd')
KINDS=('maximum_reported_number','minimum_reported_number','maximum_depth','minimum_depth','depth_change','reported_number_change')
PREFIX='<image>\n' # exact published renderer build.py source, not a broad media/text regex

def need(ok,why):
    if not ok:raise ValueError(why)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()

def convert_rows(rows):
    need(len(rows)==72 and all(isinstance(x,dict) for x in rows),'full72 source denominator required')
    ids=[x['id'] for x in rows]
    need(all(type(x) is str and x for x in ids) and len(set(ids))==72,'full72 unique original IDs required')
    charts={};groups=set();kinds=Counter();output=[];witness=[]
    for ordinal,row in enumerate(rows):
        need('labels' not in row and [m.get('role') for m in row['messages']]==['user','assistant'] and
             len(row.get('images',[]))==1 and not any(k in row for k in ('videos','audios','objects')),
             'original supervised single-image train row required')
        image=row['images'][0];p=PurePosixPath(image)
        need(p.parent==PurePosixPath(DATA+'/train/media') and p.suffix=='.png' and str(p)==image and '..' not in p.parts,
             'only exact original target train image references, never open them')
        question=row['messages'][0]['content'];answer=row['messages'][1]['content']
        need(type(question) is str and question.startswith(PREFIX) and question.count('<image>')==1,
             'exact single leading renderer image placeholder required')
        text=question[len(PREFIX):] # no strip/reformat/rewrite; all question bytes after marker retained
        need(text and not any(token in text for token in ('<image>','<video>','<audio>','<start_of_image>','<end_of_image>')),
             'unexpected media token; no permissive placeholder cleanup')
        need(type(answer) is str and answer.startswith('FINAL: ') and len(answer)>7,
             'original FINAL answer must be retained verbatim')
        kind=row['kind'];need(kind in KINDS,'original six-kind task required');kinds[kind]+=1
        groups.add(row['group']);key=row['chart_id']
        record=charts.setdefault(key,dict(image=image,group=row['group'],kinds=[]))
        need(record['image']==image and record['group']==row['group'],'same source chart/group/image identity required')
        record['kinds'].append(kind)
        value=copy.deepcopy(row);value.pop('images');value['messages'][0]['content']=text
        need(value['messages'][1]==row['messages'][1] and value['id']==row['id'],'source supervision or ID changed')
        output.append(value)
        witness.append(dict(ordinal=ordinal,source_ID_SHA256=sha(row['id'].encode()),source_row_SHA256=sha(canonical(row)),
            source_image_reference_SHA256=sha(image.encode()),question_semantics_SHA256=sha(text.encode()),
            assistant_verbatim_SHA256=sha(answer.encode()),output_row_SHA256=sha(canonical(value)),removed_prefix=PREFIX))
    need(len(charts)==12 and len({x['image'] for x in charts.values()})==12 and len(groups)==3 and
         dict(kinds)=={k:12 for k in KINDS} and all(len(x['kinds'])==6 and set(x['kinds'])==set(KINDS) for x in charts.values()),
         'original72/12chart/3group/six-kind complete coverage required')
    return output,dict(source_rows=72,source_ID_order_SHA256=sha(canonical(ids)),source_images=12,source_charts=12,
        source_groups=3,kind_counts=dict(kinds),source_image_reference_SHA256=sorted({w['source_image_reference_SHA256'] for w in witness}),
        witness=witness,output_images=0,media_bodies_read=0,blank_images_created=0,new_independent_samples=0,
        assistant_labels_unchanged=True,question_only_change='Remove images field and exact leading <image> newline only',
        source_order_unchanged=True,dev_or_blind_read=False)
