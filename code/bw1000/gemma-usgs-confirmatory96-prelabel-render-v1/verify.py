"""Independently verify the exact frozen input-only bundle without loading images into a model."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent if HERE.parent.name=='local-only' else HERE.parents[2]
OUTPUT=ROOT/'local-only/gemma-usgs-confirmatory96-prelabel-render-49253df040cf-original'
PIN='9e53e81251a722680c7d15c607f27377740fe234bd400971b9513864508e2ec7'


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def read(path,cap):
    if path.resolve(strict=True)!=path or not 0<path.stat().st_size<=cap:
        raise ValueError('EXACT_BOUNDED_PRIVATE_BUNDLE_PATH')
    return path.read_bytes()


def verify():
    raw=read(OUTPUT/'INPUT_FILES_PRIVATE.json',65536)
    assert pin(raw)['SHA256']==PIN
    mf=json.loads(raw)
    assert mf['native_model_GO'] is mf['blind_scoring_GO'] is False
    paths={p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob('*') if p.is_file()}
    chart_paths={chart['image'] for chart in mf['charts']}
    assert len(chart_paths)==16 and paths==chart_paths|{'inputs.jsonl','RECEIPT.json','INPUT_FILES_PRIVATE.json'}
    histories={}
    formats=Counter()
    for chart in mf['charts']:
        body=read(OUTPUT/chart['image'],1048576)
        assert pin(body)==chart['pin'] and body[:16]==b'\x89PNG\r\n\x1a\n\0\0\0\rIHDR'
        cols,rows,depth,color,_,_,_=struct.unpack('>IIBBBBB',body[16:29])
        assert (cols,rows,depth)==(512,512,8) and color in (2,6)
        assert chart['id'] not in histories
        histories[chart['id']]=(chart['group'],chart['image'])
        formats['RGBA' if color==6 else 'RGB']+=1
    body=read(OUTPUT/'inputs.jsonl',65536)
    assert pin(body)==mf['inputs']
    records=[json.loads(line) for line in body.splitlines()]
    assert len(records)==len({r['id'] for r in records})==96
    assert len({(r['images'][0],r['messages'][0]['content']) for r in records})==96
    assert Counter(r['kind'] for r in records)==Counter(mf['prompts_by_kind'])
    assert sorted(Counter(r['group'] for r in records).values())==[12]*8
    assert sorted(Counter(r['chart_id'] for r in records).values())==[6]*16
    for row in records:
        assert set(row)=={'id','chart_id','group','kind','event_ids','images','messages'}
        assert histories[row['chart_id']]==(row['group'],row['images'][0]) and len(row['images'])==1
        assert len(row['messages'])==1 and set(row['messages'][0])=={'role','content'}
        assert row['messages'][0]['role']=='user' and row['messages'][0]['content'].startswith('<image>\n')
    return dict(passed=True,media=16,input_rows=96,png_channels=dict(formats),manifest_SHA256=PIN,
                exact_input_only_schema=True,old_data_or_gold_opened=0,network=0,predictions=0)


if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
