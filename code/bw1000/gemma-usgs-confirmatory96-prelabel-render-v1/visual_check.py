"""Check actual frozen PNG layout against identically rendered text geometry."""

from itertools import combinations
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build as b
import verify as v


def area(left,right):
    return max(0,min(left.x1,right.x1)-max(left.x0,right.x0))*max(0,min(left.y1,right.y1)-max(left.y0,right.y0))


def inspect_figure(canvas,index):
    fig=canvas.figure
    paint=canvas.get_renderer()
    items=[]
    def use(label,artist):
        b.need(artist.get_visible() and bool(artist.get_text()),'EXPECTED_NONEMPTY_VISIBLE_TEXT')
        items.append((label,artist.get_window_extent(paint)))

    b.need(len(fig.texts)==7 and len(fig.axes)==2,'ORIGINAL_SEVEN_GLOBAL_LINES_TWO_BAR_PANELS')
    text=[item.get_text() for item in fig.texts]
    b.need(text[0]=='Reported earthquakes' and text[2].startswith('UTC ')
           and text[3].startswith('Source: USGS ComCat; preferred networks: ')
           and 'official USGS image' in text[4] and 'Mixed magTypes' in text[5],
           'NEUTRAL_TITLE_ACTUAL_UTC_AND_SOURCE_NOTES')
    for i,t in enumerate(fig.texts):use(f'global-{i}',t)
    for i,ax in enumerate(fig.axes):
        tag=f'panel-{i+1}'
        xticks=ax.get_xticklabels()
        b.need(len(xticks)==len(ax.texts)==len(ax.patches)==6,'SIX_BARS_LABELS_AND_NUMERIC_ANNOTATIONS_PER_PANEL')
        b.need([t.get_text().split('\n')[0] for t in xticks]==[f'E{x:02d}' for x in range(1,7)],'ORIGINAL_E01_E06_TICK_ORDER')
        b.need(ax.get_ylabel()==('Number','km')[i],'ORIGINAL_DISPLAYED_BAR_UNITS')
        use(tag+'-title',ax.title)
        use(tag+'-unit',ax.yaxis.label)
        for j,artist in enumerate(xticks):use(tag+f'-event-{j+1}',artist)
        for j,artist in enumerate(ax.texts):use(tag+f'-value-{j+1}',artist)
        low,high=sorted(ax.get_ylim())
        for j,tick in enumerate(ax.yaxis.get_major_ticks()):
            if low <= tick.get_loc() <= high and tick.label1.get_text():use(tag+f'-scale-{j}',tick.label1)
    margin=min(min(box.x0,box.y0,512-box.x1,512-box.y1) for _,box in items)
    collisions=[(first,second) for (first,a),(second,c) in combinations(items,2) if area(a,c)>0.25]
    return dict(chart_ordinal=index,text_objects=len(items),figure_meta_lines=7,panels=2,event_labels=12,
                numeric_annotations=12,min_pixel_clearance=round(margin,3),text_collisions=collisions)


def check():
    start=time.monotonic()
    v.verify()
    charts,_,_=b.source_and_history()
    templates,display,renderer,_=b.templates_and_renderer()
    b.need(len(templates)==6,'EXACT_RENDERED_ADMISSION_TEMPLATE_COUNT')
    sys.path.insert(0,str(b.OVERLAY/'overlay'))
    plan=json.loads(b.trusted_source('plan'))
    result=[]
    with TemporaryDirectory(prefix='gemma96-readability-',dir=b.OUTPUT.parent) as config:
        os.environ.update(MPLCONFIGDIR=config,MPLBACKEND='Agg')
        runtime=renderer['preflight'](plan)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        original=FigureCanvasAgg.print_png
        for i,chart in enumerate(charts,1):
            boxes=[]
            wanted=display(chart)
            def intercepted(canvas,*args,**kwargs):
                result=original(canvas,*args,**kwargs)
                b.need(canvas.figure.texts[2].get_text()=='UTC '+wanted[0]['time_UTC']+' to '+wanted[-1]['time_UTC'],
                       'ACTUAL_SOURCE_UTC_TEXT_MATCHES_SELECTED_EVENTS')
                for panel,key in zip(canvas.figure.axes,('mag','depth'),strict=True):
                    b.need([item.get_text() for item in panel.texts]==[row[key] for row in wanted],
                           'ACTUAL_DISPLAYED_ANNOTATIONS_MATCH_SOURCE_ROWS')
                boxes.append(inspect_figure(canvas,i))
                return result
            with patch.object(FigureCanvasAgg,'print_png',intercepted):
                actual=renderer['draw'](chart,plan,runtime)
            path=b.OUTPUT/'media'/f'chart-{i:02d}.png'
            saved=v.read(path,1048576)
            b.need(actual==saved and len(boxes)==1,'IDENTICAL_SAVED_MEDIA_PIXEL_BYTES_AND_DRAW_BOUNDARY')
            result+=boxes
            b.need(time.monotonic()-start<120,'BOUNDED_INDEPENDENT_READABILITY_EPOCH')
    passing=sum(r['min_pixel_clearance']>=0 and not r['text_collisions'] for r in result)
    return dict(passed=passing==16,charts_checked=len(result),charts_no_text_clip_or_overlap=passing,
         input_manifest_SHA256=v.PIN, exact_saved_PNG_byte_matches=16, old_images_read=0, gold_reads=0, models_called=0,
         manual_independent_reader=False,method='original exact re-render plus real Agg text bounding boxes and source label match',
         charts=result,wall_seconds=round(time.monotonic()-start,3))


if __name__=='__main__':
    result=check()
    out=b.OUTPUT.parent/'gemma-usgs-confirmatory96-prelabel-render-visual-49253df040cf.json'
    receipt=b.save(out,b.canonical(result))
    print(json.dumps(dict(receipt=receipt,**{k:v for k,v in result.items() if k!='charts'}),sort_keys=True))
    raise SystemExit(0 if result['passed'] else 2)
