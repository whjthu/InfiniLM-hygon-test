"""All load/admission/forward/cleanup checks derive from one original GPU epoch."""
import os,time
from pathlib import Path
SERVICE_SECONDS=13500
HARD_NATIVE_SECONDS=14100
def remaining():return float(os.environ["GLM_R290_GPUSTART_UNIX"])+SERVICE_SECONDS-time.time()
def check(where,min_left=0):
 if os.environ.get("GLM_R404_CANCEL_FILE")and Path(os.environ["GLM_R404_CANCEL_FILE"]).exists():raise RuntimeError("Original residency cancel: "+where)
 if remaining()<=min_left:raise TimeoutError("Original residency deadline: "+where)
def monotonic_deadline():return time.monotonic()+max(0,remaining())
