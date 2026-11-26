"""Build 16 private charts and 96 input-only prompts from saved signed source receipts."""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import runpy
import sys
from tempfile import TemporaryDirectory
import time
from types import ModuleType


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent if HERE.parent.name=='local-only' else HERE.parents[2]
S=ROOT/'local-only/gemma-usgs-confirmatory96-prospective-v1'
BODY=S/'body12-after-counts-778c0c2e-original'
OUTPUT=ROOT/'local-only/gemma-usgs-confirmatory96-prelabel-render-49253df040cf-original'
R=ROOT/'code/bw1000/training-USGS-multievent-render144-v2'
OVERLAY=ROOT/'local-only/workspace-matplotlib-overlay-completion-fsid-fix-v2'
QUAL='b0ccec6b321978cd4076174465c91a80c4bc902466b69afe6691a8a4dc0b2ff8'
SELECTION='49253df040cfb9648b1372e67816409bf6bc38e348fea359b6f71e0d34dabb5c'
STYLE='8d3277cf6f4d5d0bfa96b5bd06420331a3828ae9e83ab7d0a656a444fdf8f57c'
BODY_RECEIPT='fc28e739fcefb55efba04d038a45b259ac274f7569ab5b8b6285c84a9578db0f'
SOURCES={
    'qualify':(ROOT/'code/bw1000/gemma-usgs-confirmatory96-body12-v1/qualify.py','a6f90231b8200dbb67b7e881126f31aa43333f28a74ec9ab7aabd624fc8fb0a1'),
    'body':(ROOT/'code/bw1000/gemma-usgs-confirmatory96-prospective-v1/body_once.py','84909e12de31e3febee943c2167e47b871b95662884f603056b5e4a755eece4e'),
    'policy':(ROOT/'code/bw1000/gemma-usgs-body12-closure-audit-v1/audit.py','67048a168c2f36e13c2cb880cfffb6351433aec7b384fe921ed728fe7023cdb6'),
    'contracts':(R/'contracts.py','17a2802a78a2fb3b45a9918660774d25718433a7fb477feaf4a31987faf3e68e'),
    'draw':(R/'render.py','fd4a6af93018e629e7c047ff0a56b095e84fc38ea758c6062f05b661b8615938'),
    'plan':(R/'PLAN.json','ba76cc98ef6b6d9ef1ea832b8b21a4084952282194ca4db432f461a01e37c434'),
}


def need(ok,code):
    if not ok:
        raise ValueError(code)


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def read(path,expected,cap):
    need(path.resolve(strict=True)==path,'EXACT_CANONICAL_READ_PATH')
    with path.open('rb') as stream:raw=stream.read(cap+1)
    need(0<len(raw)<=cap and pin(raw)['SHA256']==expected,'EXACT_FROZEN_INPUT_HASH_AND_EXTENT')
    return raw


def canonical(value):return (json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()


def save(path,raw):
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return pin(raw)


def trusted_source(key):
    path,sha=SOURCES[key]
    return read(path,sha,65536)


def loaded(key):
    path=SOURCES[key][0]
    ns={'__name__':'frozen_'+key,'__file__':str(path)}
    exec(compile(trusted_source(key),str(path),'exec'),ns)
    return ns


def templates_and_renderer():
    tree=ast.parse(trusted_source('contracts'))
    questions=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='questions')
    definitions=next(n.value for n in questions.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='definitions' for t in n.targets))
    suffix=next(n.value for n in ast.walk(questions) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='suffix' for t in n.targets))
    phrase=next(n.value for n in ast.walk(questions) if isinstance(n,ast.Constant)
                and n.value==' Answer FINAL: followed by the answer.')
    suffixes={'labels':ast.literal_eval(suffix.body),'decimal2':ast.literal_eval(suffix.orelse)}
    templates=[]
    for entry in definitions.elts:
        kind,question,answer_type=[ast.literal_eval(entry.elts[i]) for i in (0,1,3)]
        templates.append(dict(kind=kind,question=question+suffixes[answer_type]+phrase,answer_type=answer_type))
    tree.body.remove(questions)
    module=ModuleType('contracts')
    exec(compile(tree,'frozen_display_and_no_questions_function','exec'),module.__dict__)
    need(module.KINDS==[t['kind'] for t in templates] and not hasattr(module,'questions'),'EXACT_SIX_TEMPLATES_NO_ANSWER_FUNCTION')
    draw=trusted_source('draw')
    old=b'June 2026: reported earthquakes'
    need(draw.count(old)==1,'ORIGINAL_RENDER_TITLE_ANCHOR')
    rendered=draw.replace(old,b'Reported earthquakes')
    previous=sys.modules.get('contracts')
    sys.modules['contracts']=module
    try:
        ns={}
        exec(compile(rendered,'neutral_title_render144_v2','exec'),ns)
    finally:
        if previous is None:sys.modules.pop('contracts',None)
        else:sys.modules['contracts']=previous
    return templates,module.displayed_rows,ns,pin(rendered)


def source_and_history():
    q=ModuleType('qualification_source')
    q.__dict__.update(loaded('qualify'))
    prospective=loaded('body')
    rule=loaded('policy')['audit_body']
    routes=prospective['inputs'](q)
    qualified=json.loads(read(BODY/'QUALIFICATION.json',QUAL,8192))
    selected=read(BODY/'PRELABEL_SELECTION_PRIVATE.json',SELECTION,32768)
    crosswalk=json.loads(read(S/'style18-reserved-identity-original-readback-v1.json',STYLE,4096))
    need(qualified['state']=='EIGHT_PROSPECTIVE_GROUPS_PRELABEL_MEDIA_AND_IDENTITY_REVIEW_PENDING'
         and qualified['planned']==qualified['terminal']==qualified['completed']==12
         and qualified['selected_charts']==16 and qualified['selected_events']==96
         and qualified['selection_private']==pin(selected) and qualified['gold_generated']==0
         and crosswalk['prospective_selection']['SHA256']==SELECTION
         and all(not value for key in ('prospective_against_prior_index_identity_overlaps',
             'prospective_against_style18_identity_overlaps') for value in crosswalk[key].values())
         and crosswalk['reserved_Style18_counts']==dict(charts=6,events=36,groups=2,media=6)
         and crosswalk['actual_gold_reads']==crosswalk['actual_old_image_reads']==0,'EXACT_STYLE18_AND_QUALIFIED_SELECTION')
    receipts=json.loads(read(BODY/'BODY_RECEIPTS.json',BODY_RECEIPT,16384))
    source={}
    body_pins={}
    for route,receipt in zip(routes,receipts,strict=True):
        code=route['query_id']
        need(receipt['query_id']==code and receipt['url']==route['body_url'] and receipt['request_attempts']==1
             and receipt['http_status']==200 and receipt['status']=='COMPLETE','EXACT_ORIGINAL_ROUTE_AND_BODY_RECEIPT')
        expected=q.C.canonical(receipt)
        need(read(BODY/(code+'-receipt.json'),pin(expected)['SHA256'],8192)==expected,'EXACT_PER_ROUTE_RECEIPT_BYTES')
        item=receipt['body']
        raw=read(BODY/(code+'.geojson'),item['SHA256'],q.CAP)
        need(pin(raw)==item==qualified['bodies'][code]['body'],'EXACT_ORIGINAL_BODY_BYTES')
        rows,summary=rule(q,raw,route)
        raw_index={item['id']:item['properties']['sources'] for item in json.loads(raw)['features']}
        need(len(rows)==len(raw_index)==receipt['body_rows'] and summary['window_eligible']==receipt['preferred_US'],
             'EXACT_COMPLETE_ROUTE_AND_ELIGIBILITY')
        body_pins[code]=item
        for row in rows:
            source[(code,row['id'])]=(row,raw_index[row['id']])
    charts=json.loads(selected)
    seen=set()
    for chart in charts:
        for index,row in enumerate(chart['rows']):
            rawrow,contributors=source[(row['route'],row['id'])]
            need(row==rawrow and row['preferred_US'] is True and row['preferred_net']=='us'
                 and row['id'] not in seen and 'us' in contributors.strip(',').split(','),'EXACT_SELECTED_ROUTE_ID_VALUES_AND_SOURCES')
            seen.add(row['id'])
            chart['rows'][index]=dict(row,sources=contributors)
    need(len(body_pins)==12 and len(source)==2807 and len(charts)==len({c['id'] for c in charts})==16
         and sorted(Counter(c['group'] for c in charts).values())==[2]*8 and len(seen)==96,'EXACT_16_CHARTS_8_GROUPS_96_EVENTS')
    metadata={}
    for key in ('original_render_metadata','original_historical_identity'):
        item=crosswalk[key]
        metadata[key]=json.loads(read(ROOT/item['path'],item['SHA256'],item['bytes']))
    style_manifest=metadata['original_render_metadata']
    old_png={v['SHA256'] for name,v in style_manifest['outputs'].items() if name.lower().endswith('.png')}
    old_png.update(metadata['original_historical_identity']['media_SHA256'])
    need(len([name for name in style_manifest['outputs'] if name.lower().endswith('.png')])==24
         and len(old_png)>=82 and all(re.fullmatch('[0-9a-f]{64}',item) for item in old_png),'EXACT_HISTORIC_METADATA_PNG_DIGESTS')
    return charts,body_pins,old_png


def inputs_for(chart,image,templates):
    return [dict(id=chart['id']+':q'+str(i),chart_id=chart['id'],group=chart['group'],kind=row['kind'],
        event_ids=['E%02d'%j for j in range(1,7)],images=[image],
        messages=[dict(role='user',content='<image>\n'+row['question'])]) for i,row in enumerate(templates)]


def build():
    start=time.monotonic()
    charts,body_pins,old_png=source_and_history()
    templates,display,renderer,renderer_pin=templates_and_renderer()
    env_outcome=read(OVERLAY/'outcome.json','d4642ac075a3681e2b13611d5c43858f591f36919ccd921afbf431447f6b0c2a',13615)
    evidence=json.loads(env_outcome)
    need(evidence['passed'] is True and evidence['job']['returncode']==0 and evidence['job']['timed_out'] is False
         and (OVERLAY/'overlay').is_dir(),'EXACT_ORIGINAL_QUALIFIED_AGG_OVERLAY')
    need('matplotlib' not in sys.modules,'FRESH_APPROVED_RENDERER_PROCESS_REQUIRED')
    sys.path.insert(0,str(OVERLAY/'overlay'))
    plan=json.loads(trusted_source('plan'))
    with TemporaryDirectory(prefix='gemma96-mpl-',dir=OUTPUT.parent) as temp:
        os.environ.update(MPLCONFIGDIR=temp,MPLBACKEND='Agg',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        runtime=renderer['preflight'](plan)
        import matplotlib
        from PIL import Image
        need(matplotlib.get_backend().lower()=='agg'
             and Path(matplotlib.__file__).resolve().is_relative_to(OVERLAY/'overlay'), 'EXACT_OVERLAY_MATPLOTLIB_AGG_PROVIDER')
        OUTPUT.mkdir(mode=0o700)
        media=OUTPUT/'media'
        media.mkdir(mode=0o700)
        chart_manifest,rows,image_digests=[],[],set()
        for index,chart in enumerate(charts,1):
            name='media/chart-%02d.png'%index
            png=renderer['draw'](chart,plan,runtime)
            with Image.open(io.BytesIO(png)) as image:
                need(image.format=='PNG' and image.size==(512,512),'EXACT_RENDERED_RGB_GEOMETRY')
                image.load()
            h=pin(png)['SHA256']
            need(h not in old_png|image_digests,'OLD_OR_NEW_IMAGE_SHA256_OVERLAP')
            image_digests.add(h)
            media_pin=save(OUTPUT/name,png)
            shown=display(chart)
            chart_manifest.append(dict(id=chart['id'],group=chart['group'],image=name,pin=media_pin,
                source_route_identities=[dict(route=r['route'],id=r['id']) for r in chart['rows']],
                displayed_rows_SHA256=pin(canonical(shown))['SHA256']))
            rows+=inputs_for(chart,name,templates)
            need(time.monotonic()-start<120,'BOUNDED_LOCAL_RENDER_WINDOW')
    by_kind=Counter(row['kind'] for row in rows)
    need(len(rows)==len({row['id'] for row in rows})==96 and len(image_digests)==16 and len(by_kind)==6
         and set(by_kind.values())=={16} and all(set(row)=={'id','chart_id','group','kind','event_ids','images','messages'}
             and len(row['messages'])==1 and row['messages'][0]['role']=='user' for row in rows),
         'EXACT_UNIQUE_INPUT_ONLY_SCHEMA_AND_DENOMINATORS')
    need(len({(row['images'][0],row['messages'][0]['content']) for row in rows})==96,'EXACT_UNIQUE_PROMPT_IMAGE_PAIRS')
    input_pin=save(OUTPUT/'inputs.jsonl',b''.join(map(canonical,rows)))
    manifest=dict(schema='GEMMA_USGS_CONFIRMATORY96_PRELABEL_INPUTS_ONLY_V1',charts=chart_manifest,inputs=input_pin,
        question_kind_and_future_parser_type={r['kind']:r['answer_type'] for r in templates},
        selection_SHA256=SELECTION,qualification_SHA256=QUAL,style18_identity_readback_SHA256=STYLE,
        exact_saved_body_pins=body_pins,source_body_receipts_SHA256=BODY_RECEIPT,
        original_renderer_SHA256=SOURCES['draw'][1],neutral_title_renderer=renderer_pin,
        original_templates_SHA256=SOURCES['contracts'][1],original_overlay_outcome=pin(env_outcome),
        renderer_versions=runtime['versions'],font_SHA256=runtime['font_SHA256'],backend='Agg',
        historic_png_SHA256_checked=len(old_png),PNG_identity_overlap=0,prompts_by_kind=dict(by_kind),
        native_model_GO=False,blind_scoring_GO=False,independent_visual_review='PENDING')
    manifest_pin=save(OUTPUT/'INPUT_FILES_PRIVATE.json',canonical(manifest))
    result=dict(passed=True,actual_groups=8,actual_PNG=16,actual_inputs=96,actual_saved_bodies=12,
        selected_rows_rebound=96,prior_PNG_SHA256_checked=len(old_png),PNG_overlap=0,inputs=input_pin,manifest=manifest_pin,
        wall_seconds=round(time.monotonic()-start,3),HTTP_gets=0,gold_reads=0,gold_generated=0,predictions=0,
        independent_visual_review='PENDING',blind_scoring_GO=False,native_model_GO=False)
    result['receipt']=save(OUTPUT/'RECEIPT.json',canonical(result))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('preflight','render'))
    options=ap.parse_args()
    if options.mode=='preflight':
        templates,_,_,renderer_pin=templates_and_renderer()
        result=dict(templates=len(templates),neutral_title_renderer=renderer_pin,actual_images=0,gold_generated=0)
    else:result=build()
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
