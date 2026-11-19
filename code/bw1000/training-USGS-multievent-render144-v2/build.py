"""Build only new chart/QA artifacts from already frozen input windows."""
import json
import os
from pathlib import Path
from contracts import canonical,sha,need,displayed_rows,questions,KINDS
from render import preflight,draw

S=Path(__file__).resolve().parent;G=S.parents[2]

def stable(path,pin):
    a=path.lstat();need(path.is_file() and not path.is_symlink() and a.st_size==pin['bytes']<=2*1024**2,'input metadata bounds')
    raw=path.read_bytes();b=path.lstat();need(all(getattr(a,k)==getattr(b,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')) and sha(raw)==pin['SHA256'],'input metadata changed');return raw

def save(path,value):
    with path.open('xb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())

def source(plan):
    base=G/plan['selected_source_root'];values={n:json.loads(stable(base/n,p)) for n,p in plan['input_pins'].items()}
    need(values['worker-result.json']['passed'] is True and values['input-windows-private.json']['passed'] is True,'actual input source not qualified')
    charts=values['input-windows-private.json']['charts'];decl=values['input-selection-declaration-private.json']
    need(len(charts)==24 and len(decl['windows'])==24,'frozen24 denominator')
    expected={c['id']:c for c in decl['windows']};events=set();groups={}
    for c in charts:
        d=expected[c['id']];need(d['group']==c['group'] and d['split']==c['split'] and d['input_rows_SHA256']==sha(canonical(c['rows'])),'frozen selected window identity differs')
        need(len(c['rows'])==6 and [r['id'] for r in c['rows']]==d['event_ids'],'original six events differ')
        need(groups.setdefault(c['group'],c['split'])==c['split'],'closure across split')
        for r in c['rows']:need(r['id'] not in events,'selected event reused');events.add(r['id'])
    need(len(events)==144 and {s:sum(c['split']==s for c in charts) for s in plan['planned_charts']}==plan['planned_charts'],'actual unique events/split denominator')
    return charts

def run(root,plan,left):
    # Protocol bytes are verified and durably frozen before any gold or image creation.
    for name,pin in plan['protocol_pins'].items():stable(S/name,pin)
    need(plan['qa_kinds']==KINDS,'exact new six-kind semantics')
    save(root/'protocol-frozen-private.json',dict(protocol_pins=plan['protocol_pins'],qa_kinds=KINDS,source_pins=plan['input_pins'],
        context=4096,greedy_cap=256,all_EOS_cap_missing_denominators=True,display_rounding='ROUND_HALF_UP_2',all_ties=True,zero_changes_retained=True))
    charts=source(plan);renderer=preflight(plan);left();data=root/'dataset';data.mkdir()
    manifest=dict(schema='USGS_MULTIEVENT_RENDER144_DATA_ONLY_V1',source_pins=plan['input_pins'],protocol_pins=plan['protocol_pins'],
        planned_charts=plan['planned_charts'],planned_QA=plan['planned_QA'],renderer=renderer,Software=plan['Software_metadata'],pixels=plan['pixels'],charts=[],outputs={},
        HTTP_calls=0,model_calls=0,training_calls=0,native_calls=0,visual_semantic_readability='PENDING_BOUNDED_HUMAN_REVIEW',Dataset_GO=False,Model_GO=False)
    streams={};completed=0;qa_completed=0
    try:
        for split in plan['planned_charts']:
            (data/split/'media').mkdir(parents=True)
            for kind in ('inputs.jsonl','labels-private.jsonl'):streams[(split,kind)]=(data/split/kind).open('xb')
        for ordinal,c in enumerate(charts):
            left();split=c['split'];image=split+'/media/%03d.png'%ordinal;png=draw(c,plan,renderer)
            with (data/image).open('xb') as f:f.write(png);f.flush();os.fsync(f.fileno())
            qs=questions(c);need(len(qs)==6 and [q['kind'] for q in qs]==KINDS,'all six tasks')
            for q in qs:
                inp={k:q[k] for k in ('id','chart_id','group','kind','event_ids')};inp.update(images=[image],messages=[dict(role='user',content='<image>\n'+q['question'])],labels=None)
                streams[(split,'inputs.jsonl')].write(canonical(inp)+b'\n');streams[(split,'labels-private.jsonl')].write(canonical(q)+b'\n')
            for kind in ('inputs.jsonl','labels-private.jsonl'):f=streams[(split,kind)];f.flush();os.fsync(f.fileno())
            qa_completed+=6;completed+=1
            manifest['charts'].append(dict(id=c['id'],group=c['group'],split=split,image=image,event_ids=[r['id'] for r in c['rows']],display_rows_SHA256=sha(canonical(displayed_rows(c))),QA=6,preferred_networks=sorted({r['preferred_net'] for r in c['rows']})))
            temp=root/'build-progress.tmp';temp.write_bytes(canonical(dict(planned_images=24,completed_images=completed,planned_QA=144,completed_QA=qa_completed)));temp.replace(root/'build-progress.json');left()
    finally:
        for f in streams.values():f.close()
    for p in data.rglob('*'):
        if p.is_file():raw=p.read_bytes();manifest['outputs'][p.relative_to(data).as_posix()]=dict(bytes=len(raw),SHA256=sha(raw))
    need(len(manifest['outputs'])==30 and sum(p['bytes'] for p in manifest['outputs'].values())<=plan['total_max_bytes'],'exact24PNG+6jsonl/cost cap')
    manifest.update(actual_images=completed,actual_QA=qa_completed,actual_split_QA={s:sum(c['split']==s for c in charts)*6 for s in plan['planned_QA']})
    need(completed==24 and qa_completed==144 and manifest['actual_split_QA']==plan['planned_QA'],'full generated split denominators')
    save(data/'manifest-private.json',manifest);return dict(passed=True,actual_images=24,actual_QA=144,manifest_SHA256=sha((data/'manifest-private.json').read_bytes()),visual_semantic_readability='PENDING',native_qualification=False)
