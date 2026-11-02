"""Read failures never authorize cancellation or another submission."""
import time
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def terminal(row):return row[2].split()[0].rstrip('+')in TERMINAL
def observe(exact,watch,save,deadline,clock=time.time,sleep=time.sleep):
 while clock()<deadline:
  try:row=exact()
  except Exception as e:
   watch.setdefault('observation_errors',[]).append(dict(at=clock(),error=repr(e)));save();sleep(3);continue
  if row:
   watch.update(actual_GPU=row[0],last_Slurm=row);save()
   if terminal(row):return row
  sleep(3)
 raise TimeoutError('Original GPU observation deadline; no replay or error-triggered cancel')
def deadline_owned_active(exact,watch,deadline,now):
 if now<deadline or not watch.get('GPU_submitted')or not watch.get('actual_GPU'):return None
 try:row=exact()
 except Exception as e:watch['deadline_state_unknown_no_cancel']=repr(e);return None
 if row and row[0]==watch['actual_GPU']and not terminal(row):return row[0]
 return None
