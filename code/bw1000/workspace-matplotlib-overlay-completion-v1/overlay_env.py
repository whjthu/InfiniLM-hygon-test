"""Per-command child environment only; never edits shared Python or os.environ."""
import os
from pathlib import Path

D=Path(__file__).resolve().parent
G=D.parents[2]
RUN=G/'local-only/workspace-matplotlib-overlay-completion-v1'
OVERLAY=RUN/'overlay'


def environment(base=None):
    value=dict(os.environ if base is None else base)
    # A fresh child resolves exact seven overlay packages before existing shared dependencies.
    value['PYTHONPATH']=str(OVERLAY)
    value.update(MPLCONFIGDIR=str(RUN/'mplconfig'),MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',
                 TMP=str(RUN/'tmp'),TEMP=str(RUN/'tmp'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return value
