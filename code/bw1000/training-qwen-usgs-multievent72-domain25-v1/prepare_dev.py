"""Small, exact existing dev36 input-only bundle; never read labels or blind assets."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath

SOURCE='local-only/training-USGS-multievent-render144-v2-original/dataset'
TARGET='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-usgs-multievent36-effect-inputs-v1'
PINS={
 'dev/inputs.jsonl':dict(bytes=21762,SHA256='25d138c9d4eff76a4c042128375e9679052f4023a0a05d322e0a6f4ad13c59e6'),
 'dev/media/012.png':dict(bytes=37939,SHA256='f3cdcf07201d184582dc1a577ed339d9651fd203dce14fdde762afd37f97195a'),
 'dev/media/013.png':dict(bytes=37833,SHA256='062787df307f8e6b87231d8abdfd93bf225c1bbf170b786959af120ee317f4a7'),
 'dev/media/014.png':dict(bytes=38564,SHA256='3ab843e170f348635cf08662f38ea0a7053cf4257a53e841d6112692e394e90b'),
 'dev/media/015.png':dict(bytes=37869,SHA256='f5f47dd9bd1914dd922a0b344efa5cfc5d63a38fafe4bedcbf123c400ba6d2a0'),
 'dev/media/016.png':dict(bytes=37195,SHA256='489fa55f8f071389ca6fd3331a631ad78af01090366370ebbbc22bd6a15cda7d'),
 'dev/media/017.png':dict(bytes=36648,SHA256='6a92e8278d6b1e7ad8d33d0b9eb22d3f56e2f760eb3bf44e6fcafbf2cf50420c'),
}


def need(ok,reason):
    if not ok:raise ValueError(reason)


def pin(raw):
    return dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())


def check_rows(raw):
    rows=[json.loads(line) for line in raw.splitlines()]
    need(len(rows)==len({r['id'] for r in rows})==36,'full unique36 dev denominator')
    known=set(PINS)-{'dev/inputs.jsonl'}
    charts=set();groups=set();counts={}
    for row in rows:
        need(row['labels'] is None and len(row['messages'])==1 and
             row['messages'][0]['role']=='user' and len(row['images'])==1 and
             row['images'][0] in known,'answer-free fixed user/image input required')
        charts.add(row['chart_id']);groups.add(row['group']);counts[row['kind']]=counts.get(row['kind'],0)+1
    from contracts import KINDS
    need(len(charts)==6 and len(groups)>=2 and set(counts)==set(KINDS) and set(counts.values())=={6},
         'whole six charts/closure groups/six-kind coverage differs')
    return rows,dict(QA=36,charts=6,closure_groups=len(groups),kind_counts=counts)


def build(repo,output):
    from PIL import Image
    source=repo/SOURCE
    data={}
    for name,expected in PINS.items():
        path=source.joinpath(*PurePosixPath(name).parts)
        need(path.is_file() and not path.is_symlink(),'exact original dev input missing')
        before=path.stat()
        with path.open('rb') as f:raw=f.read(expected['bytes']+1)
        after=path.stat()
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
             pin(raw)==expected,'actual original dev input identity changed')
        if name.endswith('.png'):
            with Image.open(io.BytesIO(raw)) as im:
                need(im.format=='PNG' and im.size==(512,512) and im.mode=='RGBA','original PNG native layout differs')
                im.load()
                need(any(lo!=hi for lo,hi in im.getextrema()[:3]),'empty/constant public chart')
        data[name]=base64.b64encode(raw).decode('ascii')
    _,summary=check_rows(base64.b64decode(data['dev/inputs.jsonl']))
    output.mkdir(mode=0o700)
    payload=dict(schema='QWEN_DEV36_INPUTS_ONLY_EXISTING_PUBLIC',target=TARGET,files=data,pins=PINS,
        labels_read=False,blind_read=False)
    raw=(json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n').encode()
    need(len(raw)<=1<<20,'bounded private input bundle')
    with (output/'dev36-inputs-private.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    result=dict(passed=True,**summary,source_files=7,source_bytes=sum(p['bytes'] for p in PINS.values()),
        PNG_decoded=6,shape=[512,512],bundle=pin(raw),dataset_gold_read=False,blind_read=False,
        root_target_staging_completed=False,native_processor_calls=0,training_updates=0,GPU_calls=0)
    (output/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf8')
    return result


def receive(payload):
    """Root runs once on HOST through its authorized existing transport, never by this module."""
    need(payload['target']==TARGET and payload['pins']==PINS and payload['labels_read'] is False and
         payload['blind_read'] is False and set(payload['files'])==set(PINS),'fixed input-only stage scope differs')
    values={}
    for name,expected in PINS.items():
        value=base64.b64decode(payload['files'][name],validate=True)
        need(pin(value)==expected,'input bundle pin differs');values[name]=value
    check_rows(values['dev/inputs.jsonl'])
    root=Path(TARGET);root.mkdir(mode=0o700)
    for name,raw in values.items():
        path=root/name;path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        need(pin(path.read_bytes())==PINS[name],'staged member differs')
    result=dict(passed=True,files=PINS,QA=36,charts=6,labels_read=False,blind_read=False,
        GPU_calls=0,model_calls=0,training_updates=0)
    with (root/'stage-complete.json').open('x',encoding='utf8') as f:json.dump(result,f,sort_keys=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.repo,a.output),sort_keys=True))
