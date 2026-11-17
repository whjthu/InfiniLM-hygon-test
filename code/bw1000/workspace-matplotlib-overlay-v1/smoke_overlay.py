"""Future child smoke only: generated toy curve, fixed package font, no data GET."""
import hashlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys

RUN=Path(os.environ['MPLCONFIGDIR']).parent
OVERLAY=(RUN/'overlay').resolve()
SOURCE=Path(__file__) if '__file__' in globals() else None
lock=json.loads((RUN/'install-lock.json').read_bytes())
modules={'matplotlib':'matplotlib','contourpy':'contourpy','cycler':'cycler','fonttools':'fontTools',
         'kiwisolver':'kiwisolver','pyparsing':'pyparsing','six':'six'}
paths={}
for name,module_name in modules.items():
    module=importlib.import_module(module_name);path=Path(module.__file__).resolve()
    assert path.is_relative_to(OVERLAY) and md.version(name)==lock['versions'][name]
    paths[name]=dict(version=md.version(name),module_path=str(path),metadata_path=str(md.distribution(name)._path))
for name,version in lock['existing_runtime'].items():
    assert md.version(name)==version
    assert not Path(md.distribution(name)._path).resolve().is_relative_to(OVERLAY)
    paths[name]=dict(version=version,metadata_path=str(md.distribution(name)._path))
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
assert font.is_file() and font.resolve().is_relative_to(OVERLAY)
figure=Figure(figsize=(4,2.5),dpi=100);FigureCanvasAgg(figure);axes=figure.subplots()
axes.plot([0,1,2],[0,1,0]);axes.set_title('Overlay smoke',fontproperties=FontProperties(fname=str(font)))
output=RUN/'smoke.png';figure.savefig(output);assert 0<output.stat().st_size<=1024**2
assert 'torch' not in sys.modules
record=dict(passed=True,paths=paths,package_font=dict(path=str(font),bytes=font.stat().st_size,
            SHA256=hashlib.sha256(font.read_bytes()).hexdigest()),backend=matplotlib.get_backend(),
            MPLCONFIGDIR=os.environ['MPLCONFIGDIR'],PNG_bytes=output.stat().st_size,WDI_GETs=0,Torch_imports=0,GPU=0)
(RUN/'smoke-result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
