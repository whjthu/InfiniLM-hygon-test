"""Bounded accounting visibility reconciliation; no Slurm command or launcher.
A present RUNNING row is not a terminal result even after the srun client returns0.
Caller owns original deadline and persists every observation, including missing rows.
"""
import time
TERMINAL=frozenset(['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'])
def is_terminal(row):
 return bool(row and len(row)>=5 and row[2].split()[0] in TERMINAL)
def wait_terminal(read,deadline,*,interval=2.,max_observations=4,monotonic=time.monotonic,sleep=time.sleep):
 if interval<=0 or max_observations<1:raise ValueError('positive bounded observation policy required')
 observations=[]
 for i in range(max_observations):
  if monotonic()>=deadline:break
  row=read();observations.append(row)
  if is_terminal(row):return dict(terminal_visible=True,row=row,observations=observations)
  if i+1<max_observations:
   remaining=deadline-monotonic()
   if remaining<=0:break
   sleep(min(interval,remaining))
 return dict(terminal_visible=False,row=None,observations=observations)
