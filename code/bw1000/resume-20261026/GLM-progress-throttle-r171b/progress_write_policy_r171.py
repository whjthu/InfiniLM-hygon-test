"""Owner-thread-only shared progress write gate. Default OFF, never changes messages/watchdog."""
import time,threading
class ProgressWritePolicy:
 def __init__(self,clock=time.monotonic):self.clock=clock;self.owner=threading.get_ident();self.configure(False)
 def check_owner(self):
  if threading.get_ident()!=self.owner:raise RuntimeError('progress policy owner thread differs')
 def configure(self,enabled):
  self.check_owner()
  if type(enabled)is not bool:raise TypeError('strict bool progress throttle')
  self.enabled=enabled;self.last_write=None;self.attempts=0;self.skipped=0;self.writes=0;self.forced_attempts=0;self.pending=False
 def ready(self,op,active,force=False):
  self.check_owner()
  if type(active)is not bool or type(force)is not bool:raise TypeError('strict progress state')
  self.attempts+=1;mandatory=force or self.last_write is None or op in ('BEGIN','DONE','ERROR')or not active
  if mandatory:self.forced_attempts+=1
  if self.enabled and not mandatory and self.clock()-self.last_write<1.0:self.skipped+=1;return False
  self.pending=True;return True
 def commit(self):
  self.check_owner()
  if not self.pending:raise RuntimeError('commit without successful write intent')
  self.last_write=self.clock();self.writes+=1;self.pending=False
 def snapshot(self):
  self.check_owner();return {'enabled':self.enabled,'routine_min_interval_seconds':1.0,'attempts':self.attempts,'skipped':self.skipped,'successful_writes':self.writes,'forced_attempts':self.forced_attempts,'owner_thread_only':True,'watchdog_seconds_unchanged':60,'background_thread':False}
