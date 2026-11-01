"""One original-start bound; source preparation never starts its clock."""
import os,time
SERVICE_SECONDS=27720
HARD_NATIVE_SECONDS=28800

def remaining():return float(os.environ['GLM_R290_GPUSTART_UNIX'])+SERVICE_SECONDS-time.time()
def check(where,min_left=0):
 if remaining()<=min_left:raise TimeoutError('Original residency deadline: '+where)
def monotonic_deadline():return time.monotonic()+max(0,remaining())
