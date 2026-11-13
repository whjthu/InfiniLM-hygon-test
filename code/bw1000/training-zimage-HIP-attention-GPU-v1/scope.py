"""One original epoch covers installation, libraries, SDK and cleanup."""
from pathlib import Path
import math,time
ROOT=Path(__file__).resolve().parent.parent
LABEL='training-zimage-HIP-attention-GPU-v1';NODE='f11r4n13';PARENT='1087656'
PREFLIGHT_SECONDS=120;INSIDE_SECONDS=420;GPU_SECONDS=480;WHOLE_SECONDS=660
TERM_SECONDS=5;KILL_SECONDS=5

def remaining(epoch,offset,now=None):
 now=time.time() if now is None else now
 if not all(math.isfinite(x) for x in (epoch,offset,now)) or epoch<=0 or not 0<offset<=WHOLE_SECONDS or not epoch<=now<epoch+offset:raise TimeoutError('original epoch exhausted')
 return epoch+offset-now

def gpu_minutes(epoch,now):
 left=remaining(epoch,GPU_SECONDS,now)
 if now>epoch+PREFLIGHT_SECONDS:raise TimeoutError('preflight expired before GPU dispatch')
 minutes=int(math.ceil(left/60))
 if minutes<1 or now+minutes*60<=epoch+INSIDE_SECONDS+TERM_SECONDS+KILL_SECONDS:raise ValueError('Slurm limit cannot cover SDK drain')
 return minutes

class EpochClock:
 def __init__(self,epoch,*,wall=time.time,monotonic=time.monotonic):
  self.epoch,self.wall,self.monotonic=epoch,wall,monotonic;self.anchor=monotonic();self.elapsed=wall()-epoch;remaining(epoch,WHOLE_SECONDS,epoch+self.elapsed)
 def left(self,offset,reserve=0):
  value=min(remaining(self.epoch,offset,self.wall()),offset-self.elapsed-(self.monotonic()-self.anchor))-reserve
  if not math.isfinite(value) or value<=0:raise TimeoutError('original monotonic epoch exhausted')
  return value
