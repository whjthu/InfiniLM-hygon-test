"""Run unchanged original capture under its original parent epoch."""
from pathlib import Path
import math,os,runpy,signal,sys,time
root=Path(__file__).resolve().parent.parent
epoch=float(os.environ['T10_EPOCH']);deadline=float(os.environ['T10_DEADLINE'])
assert math.isfinite(epoch) and epoch>0 and deadline==epoch+260 and epoch<=time.time()<deadline
def expired(*_):raise TimeoutError('original A100 capture deadline exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
assert sys.argv[1:]==[str(root/'capture')]
sys.argv=[str(root/'src/capture_a100_lora.py'),str(root/'capture')]
runpy.run_path(sys.argv[0],run_name='__main__')
