"""One snapshot -> deterministic locally rendered macro charts/QA; no original-site images."""
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.metadata as md
import io
import json
from pathlib import Path
import time

from contracts import array_response,canonical,display,need,parse_countries,parse_facts,parse_indicator,plan_check,questions,sha,strict_json


def renderer_preflight(plan):
    actual={name:md.version(name) for name in plan['renderer_versions']}
    need(actual==plan['renderer_versions'],'Exact private renderer environment missing/changed; refuse before any GET')
    import matplotlib
    matplotlib.use('Agg',force=True)
    font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    need(font.is_file() and not font.is_symlink(),'Bundled renderer font unavailable')
    font_sha=hashlib.sha256(font.read_bytes()).hexdigest()
    need(font_sha==plan['renderer_font_SHA256'],'Frozen qualified renderer font changed before firstGET')
    return dict(versions=actual,font_SHA256=font_sha,font_file=font.as_posix())


def draw_chart(country_name,indicator,years,values,plan,renderer):
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=renderer['font_file'])
    with mpl.rc_context({'text.usetex':False,'axes.formatter.use_locale':False,'path.simplify':False,'svg.hashsalt':'worldbank-macro-chart-v1'}):
        figure=Figure(figsize=(5.12,5.12),dpi=100,facecolor='white');canvas=FigureCanvasAgg(figure)
        axes=figure.add_axes((.15,.20,.81,.61))
        points=[float(Decimal(display(v))) for v in values]
        axes.plot(years,points,color='#225ea8',linewidth=1.8,marker='o',markersize=4)
        axes.set_xticks(years);axes.tick_params(axis='both',labelsize=8);axes.grid(alpha=.25)
        axes.margins(x=.08,y=.25)
        for i,(year,value) in enumerate(zip(years,points)):
            axes.annotate(display(values[i]),(year,value),xytext=(0,9 if i%2==0 else -15),textcoords='offset points',ha='center',fontproperties=font,fontsize=8)
        for label in axes.get_xticklabels()+axes.get_yticklabels():label.set_fontproperties(font)
        axes.set_ylabel(plan['indicators'][indicator]['unit_label'],fontproperties=font,fontsize=9)
        axes.set_xlabel('Year',fontproperties=font,fontsize=9)
        figure.text(.5,.94,country_name,ha='center',fontproperties=font,fontsize=12)
        figure.text(.5,.88,plan['indicators'][indicator]['name'],ha='center',fontproperties=font,fontsize=10)
        figure.text(.5,.10,'Local rendering of World Bank WDI data; CC BY 4.0',ha='center',fontproperties=font,fontsize=7)
        figure.text(.5,.065,'Displayed values: 2 decimals. Not a World Bank original chart.',ha='center',fontproperties=font,fontsize=7)
        out=io.BytesIO();canvas.print_png(out,metadata={'Software':'training-worldbank-macro-chart-v1'})
        raw=out.getvalue();need(raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw)<=plan['max_chart_bytes'],'Derived chart cap/format exceeded')
        return raw


def build(raw_documents,plan,output,check,renderer):
    plan_check(plan);need(not output.exists(),'Derived output already exists');output.mkdir()
    metadata={}
    for name,endpoint in plan['endpoints'].items():
        if 'indicator' in endpoint:metadata[endpoint['indicator']]=parse_indicator(strict_json(raw_documents[name]),endpoint['indicator'],plan)
    countries=parse_countries(strict_json(raw_documents['countries']),plan)
    facts={};versions={};nonmissing=0
    for split in ('train','dev','blind'):
        facts[split],versions[split]=parse_facts(strict_json(raw_documents['facts_'+split]),split,plan);nonmissing+=versions[split]['actual_nonmissing']
    need(len({v['lastupdated'] for v in versions.values()})==1,'Dataset vintage drift across split requests')
    prepared=[];vector_groups={}
    for split,group in plan['splits'].items():
        for country in sorted(group['countries']):
            for indicator in sorted(plan['indicators']):
                chart_id='wdi:'+country+':'+indicator+':'+str(group['years'][0])+'-'+str(group['years'][-1])
                values=[facts[split][country,indicator,year] for year in group['years']]
                value=dict(split=split,country=country,indicator=indicator,id=chart_id,years=group['years'],values=values,status='MISSING_SOURCE_VALUES')
                if all(v is not None for v in values):
                    qs,pins=questions(country,indicator,group['years'],values,chart_id,plan)
                    value.update(status='READY',questions=qs,**pins)
                    vector_groups.setdefault(pins['numeric_vector_SHA256'],[]).append(value)
                prepared.append(value)
    # Identical displayed value vectors across any country/theme/split cannot leak as duplicates.
    for values in vector_groups.values():
        if len(values)>1:
            for value in values:value['status']='DUPLICATE_DISPLAYED_VALUE_VECTOR_ALL_MEMBERS_REJECTED'
    inputs={s:[] for s in plan['splits']};labels={s:[] for s in plan['splits']};charts=[];png_hashes={};png_bytes=0
    for number,value in enumerate(prepared):
        check();split=value['split'];chart=dict(slot=number,id=value['id'],split=split,country_group=value['country'],
            indicator=value['indicator'],years=value['years'],status=value['status'])
        if value['status']=='READY':
            path=output/split/'media'/f'{number:03d}.png';path.parent.mkdir(parents=True,exist_ok=True)
            raw=draw_chart(countries[value['country']]['name'],value['indicator'],value['years'],value['values'],plan,renderer)
            digest=hashlib.sha256(raw).hexdigest();png_bytes+=len(raw);need(png_bytes<=plan['max_all_chart_bytes'],'Derived media total cap exceeded')
            need(digest not in png_hashes,'Exact derived PNG duplicated; fail without re-selection');png_hashes[digest]=number
            path.write_bytes(raw)
            chart.update(status='QUALIFIED',image=dict(name=path.relative_to(output).as_posix(),bytes=len(raw),SHA256=digest,width=512,height=512),
                numeric_vector_SHA256=value['numeric_vector_SHA256'],source_group_SHA256=sha([value['country'],value['indicator'],value['years']]),
                derived_from_official_numbers=True,original_WB_site_image=False)
            for q in value['questions']:
                item=dict(q,data_status='QUALIFIED');labels[split].append(item)
                inputs[split].append(dict(id=q['id'],messages=[dict(role='user',content='<image>\n'+q['question'])],images=[chart['image']['name']]))
        else:
            for kind in ('lookup','change','extrema'):
                for index in range(3):
                    labels[split].append(dict(id=value['id']+':'+kind+':'+str(index),kind=kind,country_group=value['country'],chart_group=value['id'],indicator=value['indicator'],
                        data_status=value['status'],question=None,answer=None,answer_type='year' if kind=='extrema' and index==2 else 'decimal2'))
        charts.append(chart)
    outputs={}
    for split in plan['splits']:
        directory=output/split;directory.mkdir(exist_ok=True)
        for name,rows in (('inputs.jsonl',inputs[split]),('labels-private.jsonl',labels[split])):
            raw=b''.join(canonical(row) for row in rows);(directory/name).write_bytes(raw)
            outputs[split+'/'+name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),rows=len(rows))
    credits=dict(dataset='World Development Indicators',publisher='The World Bank and original data providers',license='CC BY 4.0',
        license_URL='https://creativecommons.org/licenses/by/4.0/',terms_URL='https://data.worldbank.org/summary-terms-of-use',
        metadata=metadata,changes='Local charts, displayed rounding, generated questions and deterministic labels; not original World Bank figures.',
        endorsement=False,version_snapshots=versions)
    # Decimal does not appear in metadata/source descriptions, only numeric source facts.
    (output/'CREDITS-private.json').write_bytes(canonical(credits))
    counts=Counter(c['status'] for c in charts)
    manifest=dict(state='DATA_ARTIFACTS_ONLY_NATIVE_AND_MODEL_PENDING',planned_facts=504,actual_nonmissing_facts=nonmissing,missing_facts=504-nonmissing,
        planned_charts=72,actual_charts=counts['QUALIFIED'],chart_status_counts=dict(counts),planned_QA=648,
        actual_inputs=sum(len(v) for v in inputs.values()),label_rows=sum(len(v) for v in labels.values()),
        source_vintage=next(iter(versions.values()))['lastupdated'],split_country_year_disjoint=True,
        country_counts={s:8 for s in plan['splits']},questions_per_chart=9,per_kind_planned=216,
        chart_pixel_size=[512,512],actual_PNG_bytes=png_bytes,renderer=renderer,outputs=outputs,
        source_JSON_SHA256={k:hashlib.sha256(v).hexdigest() for k,v in raw_documents.items()},charts=charts,
        visual_readability='UNKNOWN_NOT_INDEPENDENTLY_REVIEWED',native_processor='NOT_RUN',numerical_training_qualification=False,
        existing_Treasury_or_formal17_inputs_changed=False,domain_checkpoint_produced=False,Dataset_GO=False,training_GO=False,GPU_GO=False)
    need(manifest['label_rows']==648 and all(len(labels[s])==216 for s in plan['splits']),'Full648 planned label slots not retained')
    (output/'manifest-private.json').write_bytes(canonical(manifest));return manifest
