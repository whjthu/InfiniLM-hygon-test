"""Qualified local renderer, new explicit two-panel layout from frozen display strings."""
import importlib.metadata as md
import io
from pathlib import Path
from contracts import need,sha,displayed_rows

def preflight(plan):
    versions={n:md.version(n) for n in plan['renderer_versions']};need(versions==plan['renderer_versions'],'qualified renderer versions changed')
    import matplotlib
    matplotlib.use('Agg',force=True)
    font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    need(font.is_file() and not font.is_symlink() and sha(font.read_bytes())==plan['font_SHA256'],'qualified font identity differs')
    return dict(versions=versions,font_file=str(font),font_SHA256=plan['font_SHA256'])

def draw(chart,plan,renderer):
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties
    rows=displayed_rows(chart);font=FontProperties(fname=renderer['font_file'])
    with mpl.rc_context({'text.usetex':False,'axes.formatter.use_locale':False,'path.simplify':False,'font.family':'DejaVu Sans'}):
        fig=Figure(figsize=(5.12,5.12),dpi=100,facecolor='white');canvas=FigureCanvasAgg(fig)
        fig.text(.5,.965,'June 2026: reported earthquakes',ha='center',fontproperties=font,fontsize=12)
        fig.text(.5,.925,'E01 to E06: chronological order; two displayed decimals',ha='center',fontproperties=font,fontsize=8)
        for position,key,title,unit,color in [((.14,.63,.82,.21),'mag','Reported magnitude NUMBER','Number','#426c90'),((.14,.27,.82,.23),'depth','Reported depth','km','#478573')]:
            ax=fig.add_axes(position);values=[float(r[key]) for r in rows];ax.bar(range(6),values,color=color)
            ax.axhline(0,color='black',linewidth=.5);ax.set_ylabel(unit,fontproperties=font,fontsize=9)
            ax.set_title(title,fontproperties=font,fontsize=10);ax.margins(y=.35);ax.tick_params(labelsize=8)
            ax.set_xticks(range(6),[r['label']+'\n'+r['magType'] if key=='mag' else r['label'] for r in rows],fontproperties=font,fontsize=8)
            for i,(v,r) in enumerate(zip(values,rows)):ax.annotate(r[key],(i,v),xytext=(0,4 if v>=0 else -11),textcoords='offset points',ha='center',fontproperties=font,fontsize=8.5)
        fig.text(.5,.175,'UTC '+rows[0]['time_UTC']+' to '+rows[-1]['time_UTC'],ha='center',fontproperties=font,fontsize=7)
        nets=', '.join(sorted({r['preferred_net'] for r in rows}))
        fig.text(.5,.13,'Source: USGS ComCat; preferred networks: '+nets,ha='center',fontproperties=font,fontsize=7)
        fig.text(.5,.095,'Locally rendered numerical data, not an official USGS image.',ha='center',fontproperties=font,fontsize=7)
        fig.text(.5,.06,'Mixed magTypes: compare numbers only, not energy or risk.',ha='center',fontproperties=font,fontsize=7)
        fig.text(.5,.025,'ROUND_HALF_UP to 2 decimals; ties and zero changes retained.',ha='center',fontproperties=font,fontsize=7)
        buffer=io.BytesIO();canvas.print_png(buffer,metadata={'Software':plan['Software_metadata']});raw=buffer.getvalue()
        need(len(raw)<=plan['PNG_max_bytes'],'PNG body cap');return raw
