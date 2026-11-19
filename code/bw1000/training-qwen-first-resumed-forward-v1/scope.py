"""One original epoch for preflight, native training, Slurm and cleanup."""
import math,os,time
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
PHASE=os.environ.get('QWEN_PAIR_PHASE')
if PHASE not in (None, 'continuous', 'resumed'):
 raise ValueError('Unexpected Qwen paired run phase')
ROOT=BASE if PHASE is None else BASE/PHASE
LABEL='training-qwen-first-resumed-forward-v1'
NODE='a01r4n16';PARENT='1087656';PORT=29979
PREFLIGHT_SECONDS=390
NATIVE_SECONDS=1350
INSIDE_SECONDS=1410
GPU_SECONDS=1500
WHOLE_SECONDS=1740
POST_SECONDS=120
TERM_SECONDS=10
KILL_SECONDS=10

def remaining(epoch,offset,*,now=None):
 now=time.time() if now is None else now
 if not all(math.isfinite(x) for x in (epoch,offset,now)) or epoch<=0 or not 0<offset<=WHOLE_SECONDS or not epoch<=now<epoch+offset:
  raise TimeoutError('original scope epoch is invalid or exhausted')
 return epoch+offset-now

def gpu_minutes(epoch,now):
 left=remaining(epoch,GPU_SECONDS,now=now)
 if now>epoch+PREFLIGHT_SECONDS:raise TimeoutError('preflight did not fit the original epoch')
 minutes=int(left//60)
 if minutes<1 or now+60*minutes<=epoch+INSIDE_SECONDS:raise ValueError('Slurm cap cannot cover native cleanup')
 return minutes

class EpochClock:
 def __init__(self,epoch,*,wall=time.time,monotonic=time.monotonic):
  self.epoch,self.wall,self.monotonic=epoch,wall,monotonic
  self.started=monotonic();self.elapsed_at_start=wall()-epoch
  remaining(epoch,WHOLE_SECONDS,now=epoch+self.elapsed_at_start)
 def left(self,offset,reserve=0):
  wall_left=remaining(self.epoch,offset,now=self.wall())
  mono_left=offset-self.elapsed_at_start-(self.monotonic()-self.started)
  value=min(wall_left,mono_left)-reserve
  if not math.isfinite(value) or value<=0:raise TimeoutError('original monotonic scope deadline exhausted')
  return value
