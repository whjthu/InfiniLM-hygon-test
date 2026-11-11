"""Pure scope/deadline/accounting/ownership checks; no Torch, Slurm or device."""
import ast,hashlib,json,math,sys
from pathlib import Path
from scope import *
from slurm_accounting import wait_terminal
from owned_processes import native_command

def verify(root):
 assert NATIVE_SECONDS+2*TERM_SECONDS<INSIDE_SECONDS<GPU_SECONDS<WHOLE_SECONDS
 assert GPU_SECONDS+40+POST_SECONDS+20<WHOLE_SECONDS
 for elapsed in (0,.001,59.999,60,179.999,180):
  epoch=1000000;now=epoch+elapsed;minutes=gpu_minutes(epoch,now)
  assert epoch+INSIDE_SECONDS<now+60*minutes<=epoch+GPU_SECONDS
 rejected=0
 for elapsed in (-1,180.001,7380,float('nan')):
  try:gpu_minutes(1000000,1000000+elapsed)
  except (ValueError,TimeoutError):rejected+=1
  else:raise AssertionError(elapsed)
 wall=[1000.];mono=[20.];clock=EpochClock(1000.,wall=lambda:wall[0],monotonic=lambda:mono[0]);wall[0]=1001.;mono[0]=25.;assert clock.left(10)==5
 wall[0]=1000.5;mono[0]=31.
 try:clock.left(10)
 except TimeoutError:pass
 else:raise AssertionError('wall rewind renewed native budget')
 sequence=iter([None,['1087656.1','test','RUNNING','0:0','1'],['1087656.1','test','COMPLETED','0:0','2']]);seen=wait_terminal(lambda:next(sequence),100,monotonic=lambda:0,sleep=lambda _:None,max_observations=3);assert seen['terminal_visible'] and len(seen['observations'])==3
 seen=wait_terminal(lambda:['1087656.1','test','RUNNING','0:0','1'],100,monotonic=lambda:0,sleep=lambda _:None,max_observations=3);assert not seen['terminal_visible'] and seen['row'] is None
 assert native_command([b'python',b'--receipt',str(ROOT/'entry-receipt.jsonl').encode()],ROOT)
 assert not native_command([b'python',b'--receipt',b'/foreign/entry-receipt.jsonl'],ROOT)
 controller=(root/'controller.py').read_text();native=(root/'native.py').read_text();install=(root/'receive_once.py').read_text()
 assert hashlib.sha256((root/'snapshot.py').read_bytes()).hexdigest()=='7fc32a4933a200d90851f460121603fcc3dd69f9fee596f47141843d3b2dbee8'
 assert "before['full_fd_inventory_performed'] is True" in native and "leased['full_fd_inventory_performed'] is False" in native
 assert 'wait_terminal(' in controller and 'terminal_visible' in controller and 'srun' in controller and "dcu:8" in controller
 assert "gpu_minutes(epoch,time.time())" in controller and "clock.left(GPU_SECONDS" in controller
 assert 'NATIVE_SECONDS' in native and 'EpochClock(epoch)' in native and "FIRST_UPDATE_EPOCH=str(epoch)" in native
 assert 'epoch=time.time();deadline=epoch+7800' in install and 'start_new_session=True' in install
 assert "'--diagnostic-callback-position','before'" in native and 'original_first_updates' in native
 assert hashlib.sha256((root/'original-launch.json').read_bytes()).hexdigest()=='015401ce4791ec978f645afd8eea7724a57f075b6d06a6eac474123418a78789'
 return dict(passed=True,valid_Slurm_deadline_points=6,rejected_deadlines=rejected,monotonic_rewind_rejected=True,actual_terminal_helper_integrated=True,missing_RUNNING_terminal_boundaries=2,exact_ownership_commands=2,source_AST_files=sum(1 for p in root.glob('*.py') if ast.parse(p.read_text())),Native_seconds=NATIVE_SECONDS,inside_seconds=INSIDE_SECONDS,GPU_seconds=GPU_SECONDS,whole_seconds=WHOLE_SECONDS,GPU_executed=False)
if __name__=='__main__':print(json.dumps(verify(Path(__file__).parent)))
