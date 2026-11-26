"""Freeze or install the exact Gemma confirmatory96 inputs-only transport objects."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import struct


HERE=Path(__file__).resolve().parent
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
LABEL='training-gemma-usgs-confirmatory96-7958f6223b96-v1'
TARGET=Path(R+'/incoming/'+LABEL+'-inputs-only')
PRIVATE_SHA='9e53e81251a722680c7d15c607f27377740fe234bd400971b9513864508e2ec7'
INPUT_PIN={'bytes':56784,'SHA256':'63962a41f124a50e1a2b38fa513cc2aaefe4dcb632162ba06d2070a462baa408'}
PREREG_SHA='b5a9a308dd50c2a940da9f9eb2a6b2bfc23872cecd37ab0b5a07816d6790f1a6'
NAMES={'inputs.jsonl',*(f'media/chart-{x:02d}.png' for x in range(1,17))}
KINDS={'maximum_reported_number','minimum_reported_number','maximum_depth','minimum_depth','depth_change','reported_number_change'}
FIELDS={'id','chart_id','group','kind','event_ids','images','messages'}


def need(flag,why):
    if not flag:raise ValueError(why)


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def encode(doc):return (json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def read(root,name,cap):
    path=root/name
    need(root.is_absolute() and path.resolve(strict=True)==path and path.is_file(),'EXACT_CANONICAL_NONSYMLINK_INPUT_MEMBER')
    with path.open('rb') as stream:raw=stream.read(cap+1)
    need(0<len(raw)<=cap,'BOUNDED_INPUT_ONLY_MEMBER')
    return raw


def exact_objects(root,manifest):
    need(set(manifest['objects'])==NAMES and manifest['inputs']==INPUT_PIN and manifest['questions']==96
         and manifest['charts']==16 and manifest['event_groups']==8 and manifest['preregistration_SHA256']==PREREG_SHA
         and manifest['private_prelabel_manifest_SHA256']==PRIVATE_SHA and manifest['gold_files_opened']==0,
         'EXACT_NATIVE_INPUT_MANIFEST_FIELDS')
    need({p.relative_to(root).as_posix() for p in root.rglob('*')}==NAMES|{'media','INPUT_FILES.json'},
         'EXACT_INPUT_ONLY_MEMBER_SET')
    files={name:read(root,name,1<<20) for name in NAMES}
    need({name:pin(raw) for name,raw in files.items()}==manifest['objects'],'EXACT_17_ANSWER_FREE_MEMBER_PINS')
    for name in NAMES-{'inputs.jsonl'}:
        raw=files[name]
        need(raw[:16]==b'\x89PNG\r\n\x1a\n\0\0\0\rIHDR' and struct.unpack('>II',raw[16:24])==(512,512),'EXACT_INPUT_IMAGE_GEOMETRY')
    rows=[json.loads(line) for line in files['inputs.jsonl'].splitlines()]
    chart={}
    kinds=Counter()
    for row in rows:
        need(set(row)==FIELDS and row['event_ids']==[f'E{x:02d}' for x in range(1,7)]
             and row['kind'] in KINDS and len(row['images'])==1 and row['images'][0] in NAMES-{'inputs.jsonl'}
             and len(row['messages'])==1 and set(row['messages'][0])=={'role','content'}
             and row['messages'][0]['role']=='user' and row['messages'][0]['content'].startswith('<image>\n'),
             'ONLY_ACTUAL_USER_IMAGE_INPUT_NO_ANSWER_FIELDS')
        identity=(row['group'],row['images'][0])
        need(chart.setdefault(row['chart_id'],identity)==identity,'EXACT_CHART_ASSIGNMENT')
        kinds[row['chart_id'],row['kind']]+=1
    need(len(rows)==len({r['id'] for r in rows})==96 and len(chart)==16
         and len({image for _,image in chart.values()})==16
         and kinds==Counter((c,k) for c in chart for k in KINDS)
         and sorted(Counter(group for group,_ in chart.values()).values())==[2]*8,
         'EXACT_8_GROUPS_16_CHARTS_96_QUESTIONS')
    return rows,files


def save(path,raw):
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def verify(root,expected):
    raw=read(root,'INPUT_FILES.json',65536)
    if isinstance(expected,str):need(pin(raw)['SHA256']==expected,'EXACT_INPUTS_ONLY_TRANSPORT_MANIFEST')
    else:need(pin(raw)==expected,'EXACT_INPUTS_ONLY_TRANSPORT_MANIFEST')
    rows,files=exact_objects(root,json.loads(raw))
    return rows,files,raw


def prepare(original,output):
    raw=read(original,'INPUT_FILES_PRIVATE.json',65536)
    need(pin(raw)['SHA256']==PRIVATE_SHA,'EXACT_ORIGINAL_PRIVATE_PRELABEL_BUNDLE')
    manifest=json.loads(raw)
    need(manifest['inputs']==INPUT_PIN and len(manifest['charts'])==16 and manifest['blind_scoring_GO'] is False,
         'ORIGINAL_AUDITED_PRELABEL_FIELDS')
    objects={chart['image']:chart['pin'] for chart in manifest['charts']}
    objects['inputs.jsonl']=INPUT_PIN
    need(set(objects)==NAMES,'EXACT_INPUT_DATA_ALLOWLIST')
    bodies={key:read(original,key,1<<20) for key in objects}
    need({key:pin(value) for key,value in bodies.items()}==objects,'EXACT_ACTIVE_PRIVATE_INPUT_DATA')
    transfer=encode(dict(objects=objects,inputs=INPUT_PIN,questions=96,charts=16,event_groups=8,
        preregistration_SHA256=PREREG_SHA,private_prelabel_manifest_SHA256=PRIVATE_SHA,gold_files_opened=0))
    output.mkdir(mode=0o700)
    (output/'media').mkdir(mode=0o700)
    for key,body in bodies.items():save(output/key,body)
    save(output/'INPUT_FILES.json',transfer)
    verify(output,pin(transfer))
    return dict(input_manifest=pin(transfer),questions=96,media=16,bytes=sum(map(len,bodies.values())),gold=0)


def install(source,expected):
    need(os.getuid()==154459 and TARGET.parent.resolve(strict=True)==TARGET.parent,'EXACT_AUTHENTICATED_INSTALL_HOST')
    rows,files,mf=verify(source,expected)
    os.umask(0o077)
    TARGET.mkdir(mode=0o700)
    (TARGET/'media').mkdir(mode=0o700)
    for key,body in files.items():save(TARGET/key,body)
    save(TARGET/'INPUT_FILES.json',mf)
    verify(TARGET,pin(mf))
    return dict(input_manifest=pin(mf),questions=len(rows),media=16,no_gpu_no_gold=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','verify','install'))
    p.add_argument('source',type=lambda v:Path(v).resolve())
    p.add_argument('output_or_manifest')
    a=p.parse_args()
    if a.mode=='prepare':answer=prepare(a.source,Path(a.output_or_manifest).resolve())
    elif a.mode=='install':answer=install(a.source,a.output_or_manifest)
    else:
        rows,files,mf=verify(a.source,a.output_or_manifest)
        answer=dict(input_manifest=pin(mf),questions=len(rows),objects=len(files),gold=0)
    print(json.dumps(answer,sort_keys=True))


if __name__=='__main__':main()
