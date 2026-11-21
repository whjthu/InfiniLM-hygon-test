"""Apply the existing direct-dispatch regression suite to the distinct retained28 scope."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import dispatch_kimi_r461_retained28_once as retained


fixture = runpy.run_path(str(Path(__file__).with_name("test_dispatch_kimi_r461_output_residual_wave_once.py")))
KimiRetained28DispatchTest = fixture["KimiOriginalScoreDispatchTest"]
KimiRetained28DispatchTest.setUp.__globals__["launcher"] = SimpleNamespace(
    **{**retained._WAVE, "WORKER": Path(retained.WORKER["path"]).parent.as_posix()})
