"""Pure offline local derived charts. Rendering is a future single bounded Job."""
from collections import Counter
from decimal import Decimal
import importlib.metadata as md
import io
from pathlib import Path
from contracts import canonical,displayed_rows,make_plan,need,questions,sha


def renderer_preflight(plan):
    actual={n:md.version(n) for n in plan['renderer_versions']};need(actual==plan['renderer_versions'],'Qualified renderer versions differ')
    import matplotlib
    matplotlib.use('Agg',force=True)
    font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    need(font.is_file() and not font.is_symlink() and sha(font.read_bytes())==plan['font_SHA256'],'Qualified font differs')
    return dict(versions=actual,font_file=font.as_posix(),font_SHA256=plan['font_SHA256'],pixels=[512,512],dpi=100,Software=plan['Software_metadata'])

def draw(chart,plan,renderer):
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=renderer['font_file']);rows=displayed_rows(chart)
    with mpl.rc_context({'text.usetex':False,'axes.formatter.use_locale':False,'path.simplify':False,'font.family':'DejaVu Sans'}):
        f=Figure(figsize=(5.12,5.12),dpi=100,facecolor='white');canvas=FigureCanvasAgg(f)
        f.text(.5,.96,'USGS reported-event sample',ha='center',fontproperties=font,fontsize=13)
        f.text(.5,.915,'Origin time UTC, Oct 2026; depths in km',ha='center',fontproperties=font,fontsize=10)
        ax=f.add_axes((.04,.49,.92,.35));ax.axis('off')
        cells=[[r['label'],r['date_UTC'],r['reported_magnitude'],r['magType'],r['depth_km']] for r in rows]
        table=ax.table(cellText=cells,colLabels=['Event','UTC MM-DD hh:mm:ss','Mag.','Type','Depth km'],colWidths=[.12,.40,.13,.14,.21],loc='center',cellLoc='center')
        table.auto_set_font_size(False);table.set_fontsize(8.5);table.scale(1,1.8)
        for cell in table.get_celld().values():cell.get_text().set_fontproperties(font);cell.get_text().set_fontsize(8.5)
        bars=f.add_axes((.16,.18,.79,.27));depths=[float(Decimal(r['depth_km'])) for r in rows]
        bars.bar([r['label'] for r in rows],depths,color='#336b87');bars.axhline(0,color='black',linewidth=.5);bars.set_ylabel('Depth (km)',fontproperties=font,fontsize=9)
        bars.tick_params(labelsize=8);bars.margins(y=.3)
        for i,r in enumerate(rows):bars.annotate(r['depth_km'],(i,depths[i]),xytext=(0,5 if depths[i]>=0 else -11),textcoords='offset points',ha='center',fontproperties=font,fontsize=8)
        f.text(.5,.10,'Locally rendered USGS numeric data; not an official USGS image.',ha='center',fontproperties=font,fontsize=7)
        f.text(.5,.065,'Two displayed decimals; magnitude types differ. Not energy/risk.',ha='center',fontproperties=font,fontsize=7)
        f.text(.5,.03,'At most first 50 filtered events; whole-week completeness unknown.',ha='center',fontproperties=font,fontsize=7)
        buffer=io.BytesIO();canvas.print_png(buffer,metadata={'Software':plan['Software_metadata']});raw=buffer.getvalue()
        need(raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw)<=plan['PNG_per_file_cap_bytes'],'Derived PNG size/format differs');return raw

def build(rows,plan,output,remaining,progress):
    need(not output.exists(),'Derived directory already exists');p=make_plan(rows,plan);output.mkdir()
    # Persist split/group plan before any question/gold or chart is constructed.
    frozen=dict(p);frozen['charts']=[{k:v for k,v in c.items() if k!='rows'} for c in p['charts']]
    (output/'split-plan-private.json').write_bytes(canonical(frozen));progress('split_frozen',dict(source_slots=len(rows),groups=p['closure_groups'],planned_charts=p['planned_charts'],planned_QA=p['planned_QA']))
    renderer=renderer_preflight(plan);inputs={s:[] for s in ['train','dev','blind']};labels={s:[] for s in inputs};records=[];hashes={};total=0
    for i,c in enumerate(p['charts']):
        remaining();qs=questions(c);record={k:v for k,v in c.items() if k!='rows'};record.update(slot=i,status='PENDING',planned_QA=5)
        try:
            raw=draw(c,plan,renderer);digest=sha(raw);need(digest not in hashes,'Duplicate rendered media; no backfill');hashes[digest]=i
            need(total+len(raw)<=plan['PNG_total_cap_bytes'],'All media cap exceeded');total+=len(raw)
            name=c['split']+'/media/'+str(i).zfill(3)+'.png';path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            record.update(status='QUALIFIED',image=dict(name=name,bytes=len(raw),SHA256=digest,width=512,height=512),display_rows_SHA256=sha(canonical(displayed_rows(c))))
            for q in qs:
                labels[c['split']].append(dict(q,data_status='QUALIFIED'))
                inputs[c['split']].append(dict(id=q['id'],chart_group=c['id'],source_group=c['group'],messages=[dict(role='user',content='<image>\n'+q['question'])],images=[name]))
        except Exception as e:
            record.update(status='FAILED',error_type=type(e).__name__,error=str(e))
            for q in qs:labels[c['split']].append(dict(q,data_status='FAILED',answer=None))
        records.append(record);progress('chart_terminal',dict(slot=i,status=record['status'],completed_chart_slots=len(records),planned_charts=p['planned_charts'],completed_QA_slots=sum(len(x) for x in labels.values()),planned_QA=p['planned_QA']))
        (output/'chart-progress-private.json').write_bytes(canonical(dict(planned_charts=p['planned_charts'],planned_QA=p['planned_QA'],completed=records)))
    outputs={}
    for split in inputs:
        d=output/split;d.mkdir(exist_ok=True)
        for name,values in [('inputs.jsonl',inputs[split]),('labels-private.jsonl',labels[split])]:
            raw=b''.join(canonical(v) for v in values);(d/name).write_bytes(raw);outputs[split+'/'+name]=dict(bytes=len(raw),SHA256=sha(raw),rows=len(values))
    credits=dict(publisher='U.S. Geological Survey Earthquake Catalog and source networks retained in private source records',source='https://earthquake.usgs.gov/fdsnws/event/1/',rights='USGS-produced numerical data: U.S. public domain/free use per official FAQ; attribution retained, no claim of worldwide CC0 or third-party media rights',rights_source='https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted',changes='Locally rendered 512x512 PNG; Decimal display rounding; deterministic reported-number QA and grouped splits; not official USGS charts.',body_SHA256=plan['original_source_body_SHA256'],api_version=plan['source_api_version'],endorsement=False)
    (output/'CREDITS.json').write_bytes(canonical(credits))
    count=Counter(r['status'] for r in records)
    manifest=dict(state='OFFLINE_DERIVED_DATA_ONLY',source_slots=len(rows),source_alias_groups=p['source_alias_groups'],source_geographic_groups=p['source_geographic_groups'],closure_groups=p['closure_groups'],split_group_counts=p['split_group_counts'],
        planned_charts=p['planned_charts'],planned_QA=p['planned_QA'],actual_charts=count['QUALIFIED'],failed_charts=count['FAILED'],actual_inputs=sum(len(v) for v in inputs.values()),retained_QA_slots=sum(len(v) for v in labels.values()),
        chart_status_counts=dict(count),charts=records,outputs=outputs,actual_PNG_bytes=total,renderer=renderer,source_pins=plan['source_pins'],split_plan_SHA256=sha((output/'split-plan-private.json').read_bytes()),credits_SHA256=sha((output/'CREDITS.json').read_bytes()),
        source_all50_covered_once=True,group_closure_before_QA=True,no_event_cross_split=True,source_window_complete='UNKNOWN_NOT_CLAIMED',independent_scene_generalization='NOT_ESTABLISHED',visual_readability='PENDING_INDEPENDENT_TRAIN_DEV_REVIEW',native_processor='NOT_RUN',Dataset_GO=False,training_GO=False,GPU_GO=False)
    need(manifest['retained_QA_slots']==p['planned_QA'],'Failure denominator not retained');(output/'manifest-private.json').write_bytes(canonical(manifest));return manifest
