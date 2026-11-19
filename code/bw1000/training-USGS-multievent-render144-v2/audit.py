"""Separate implementation: saved math, PNG bytes, split/media/field identities only."""
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import json
from contracts import need,canonical,sha

def check(root,plan,left):
    from PIL import Image
    from build import source
    # Reuse ONLY immutable source-binding checks, not producer questions/display arithmetic.
    charts=source(plan);data=root/'dataset';m=json.loads((data/'manifest-private.json').read_bytes())
    protocol=json.loads((root/'protocol-frozen-private.json').read_bytes())
    need(protocol['protocol_pins']==plan['protocol_pins']==m['protocol_pins'] and m['source_pins']==plan['input_pins'],'source/protocol binding differs')
    actual={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()};need(actual==set(m['outputs'])|{'manifest-private.json'},'exact output inventory')
    for name,pin in m['outputs'].items():
        p=data/name;raw=p.read_bytes();need(not p.is_symlink() and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'saved output pin');left()
    compare_v1_six_jsonl(root,plan,left)
    byid={c['id']:c for c in charts};allq=set();groups={};images=0;math_rows=0;kinds=Counter();zero_changes=0;ties=0;split_counts={}
    def shown(v):
        x=Decimal(str(v)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
        return Decimal(0) if x==0 else x
    for split,denom in plan['planned_QA'].items():
        ins=[json.loads(x) for x in (data/split/'inputs.jsonl').read_bytes().splitlines()];gold=[json.loads(x) for x in (data/split/'labels-private.jsonl').read_bytes().splitlines()]
        need(len(ins)==len(gold)==denom,'actual split QA denominator')
        for inp,q in zip(ins,gold):
            need(inp['id']==q['id'] and q['id'] not in allq,'ordered unique QA identity');allq.add(q['id']);c=byid[q['chart_id']]
            need(c['split']==split and q['group']==c['group'] and groups.setdefault(c['group'],split)==split,'closure split differs')
            need(inp['labels'] is None and len(inp['messages'])==1 and inp['messages'][0]['role']=='user' and inp['messages'][0]['content']=='<image>\n'+q['question'],'user-only full inputs/gold separation')
            need(inp['kind']==q['kind'] and inp['chart_id']==q['chart_id'] and inp['group']==q['group'] and inp['event_ids']==q['event_ids']==['E%02d'%i for i in range(1,7)],'task schema/labels differ')
            mags=[shown(r['values']['magnitude']) for r in c['rows']];depths=[shown(r['values']['depth_km']) for r in c['rows']]
            displayed=[dict(label='E%02d'%(i+1),time_UTC=datetime.fromtimestamp(r['time_UTC_epoch_ms']/1000,timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                mag=format(mags[i],'.2f'),depth=format(depths[i],'.2f'),magType=r['magType'],preferred_net=r['preferred_net'],sources=r['sources']) for i,r in enumerate(c['rows'])]
            display_sha=sha(canonical(displayed));mc=next(v for v in m['charts'] if v['id']==c['id'])
            need(q['display_rows_SHA256']==mc['display_rows_SHA256']==display_sha,'saved displayed-value identity differs')
            kind=q['kind'];expected=None
            if kind in ('maximum_reported_number','minimum_reported_number','maximum_depth','minimum_depth'):
                vals=mags if 'reported' in kind else depths;extreme=max(vals) if kind.startswith('maximum') else min(vals)
                expected=' '.join('E%02d'%(i+1) for i,v in enumerate(vals) if v==extreme);ties+=len(expected.split())>1
                need(q['answer_type']=='labels','label-set task type')
            else:
                need(kind in ('depth_change','reported_number_change') and q['answer_type']=='decimal2','closed six-kind registry')
                vals=depths if kind=='depth_change' else mags;delta=vals[-1]-vals[0];expected=format(Decimal(0) if delta==0 else delta,'.2f');zero_changes+=delta==0
            need(q['answer']==expected and q['data_status']=='QUALIFIED','independent displayed Decimal/ties/zero gold mismatch')
            need(inp['images']==[next(v['image'] for v in m['charts'] if v['id']==c['id'])],'exact image linkage')
            math_rows+=1;kinds[kind]+=1;left()
        split_counts[split]=len(ins)
    for c in m['charts']:
        path=data/c['image']
        with Image.open(path) as im:need(im.format=='PNG' and list(im.size)==[512,512] and im.info.get('Software')==plan['Software_metadata'],'PNG format/dim/Software');im.verify()
        with Image.open(path) as im:need(any(lo<hi for lo,hi in im.convert('RGB').getextrema()),'blank PNG')
        images+=1;left()
    need(math_rows==144 and images==24 and len(allq)==144 and len(kinds)==6 and set(kinds.values())=={24},'complete independent denominator')
    need({s:sum(v==s for v in groups.values()) for s in plan['planned_charts']}=={'train':3,'dev':2,'blind':2},'original frozen 3/2/2 closure groups')
    return dict(passed=True,independent_math_rows=144,PNG_byte_dimension_structure_nonblank_pass=24,QA_split_counts=split_counts,kind_counts=dict(kinds),
        v1_six_jsonl_byte_identical=6,old_gold_semantic_parsing=False,prior_v1_visual_failure_retained=True,
        all_ties_retained_tasks=ties,zero_change_tasks_retained=zero_changes,closure_groups={'train':3,'dev':2,'blind':2},
        visual_semantic_OCR_independent_qualification=False,visual_human_review='PENDING',native_qualification=False,training_GO=False,GPU_GO=False,Model_GO=False)

def compare_v1_six_jsonl(root,plan,left):
    from pathlib import Path
    from build import stable,G
    prior=plan['prior_visual_failure']['original_dataset_manifest']
    manifest=json.loads(stable(G/prior['path'],prior));base=(G/prior['path']).parent
    need(len(plan['v1_six_jsonl'])==6,'exact prior six JSONL denominator')
    for name,pin in plan['v1_six_jsonl'].items():
        need(manifest['outputs'][name]==pin and name.endswith('.jsonl'),'prior metadata JSONL identity differs')
        old=stable(base/name,pin);new=stable(root/'dataset'/name,pin)
        need(old==new,'layout-only repair changed raw input/label JSONL bytes')
        # No json.loads of original v1 input/gold bodies: comparison witness only.
        left()
