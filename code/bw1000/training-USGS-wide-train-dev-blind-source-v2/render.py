"""Pure offline local derived charts. Rendering is a future single bounded Job."""
from collections import Counter
from decimal import Decimal
import importlib.metadata as md
import io
from pathlib import Path
from chart_contracts import canonical,displayed_rows,need,questions,sha


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
        f.text(.5,.915,'Origin time UTC, Jan 2020; depths in km',ha='center',fontproperties=font,fontsize=10)
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
        f.text(.5,.03,'Independent grouped sample; catalog completeness unknown.',ha='center',fontproperties=font,fontsize=7)
        buffer=io.BytesIO();canvas.print_png(buffer,metadata={'Software':plan['Software_metadata']});raw=buffer.getvalue()
        need(raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw)<=plan['PNG_per_file_cap_bytes'],'Derived PNG size/format differs');return raw
