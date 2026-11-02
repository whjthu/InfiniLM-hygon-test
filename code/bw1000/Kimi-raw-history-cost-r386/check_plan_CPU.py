"""Actual fullphase consumer, recording Session/Model leaves; no Torch/Slurm."""
import copy,json,threading
from phase_plan import plan,expected,run_phase

class RecordingContext:
 def __init__(self,on):self.raw=self.projected=None;self.length=0;self.phase='closed';self.on=on
 def payload_bytes(self):return dict(raw=0 if self.on else self.length*35840*2,projected=self.length*7168*2,published_raw_input_bytes=self.length*35840*2)
class RecordingSession:
 def __init__(self,on,fail_query=None):self.context=RecordingContext(on);self.closed=False;self.sequence=0;self.q=0;self.fail_query=fail_query
 def close(self):self.closed=True;self.context.raw=self.context.projected=None
 def dispatch(self,m,b):
  assert m['seq']==self.sequence;self.sequence+=1;c=self.context;cmd=m['cmd']
  if cmd=='begin':assert c.projected is None;c.phase='prompt';c.length=0
  elif cmd.startswith('append_'):
   assert m['past']==c.length and len(b)==m['rows']*2
   if cmd=='append_committed':assert m['target_commit']==dict(capture_epoch=m['capture_epoch'],past_before=c.length,past_after=c.length+m['rows'],accepted_inputs=m['rows'],complete=True)
   c.length+=m['rows'];c.projected=True;c.raw=None if c.on else True
  elif cmd=='seal':c.phase='decode'
  elif cmd=='query':
   self.q+=1
   if self.q==self.fail_query:raise RuntimeError('CPU recorded query failure')
   return dict(mode='draft',proposals=list(range(7)),handler_wall_s=.01)
  elif cmd=='retire':c.raw=c.projected=None;c.phase='retired';return dict(removed=True)
  elif cmd=='close':self.close();return dict(closed=True)
  else:raise AssertionError(cmd)
  return dict(seq=m['seq'])

def main():
 specs=plan();records=[]
 for s in specs:records.append(run_phase(RecordingSession(s['discard_raw']),lambda past,n:bytes(n*2),s))
 assert len(records)==20 and sum(len(r['queries'])for r in records)==160 and sum(len(r['commands'])for r in records)==1190
 assert sum(sum(t['cmd']=='append_committed'for t in r['commands'])for r in records)==140
 assert sum(expected(s)['fences']for s in specs)==1270
 for L in [8635,32638]:
  a=[r for r in specs if r['L']==L];assert [r['arm']for r in a]==['OFF','ON','OFF','ON','ON','OFF','OFF','ON','ON','OFF']
  assert sum(r['phase']=='warm'for r in a)==2
 saved=[];bad=RecordingSession(False,fail_query=3)
 try:run_phase(bad,lambda past,n:bytes(n*2),specs[0],save=lambda q:saved.append(copy.deepcopy(q)))
 except RuntimeError:pass
 else:raise AssertionError('Query failure continued')
 assert bad.closed and saved[-1]['completed']and not saved[-1]['passed']and len(saved[-1]['queries'])==2
 print(json.dumps(dict(completed=True,passed=True,GPU=0,Slurm_jobs=0,Native_dlopen=False,Model_loads=0,actual_consumer='phase_plan.run_phase',recording_leaves='Session/context/query only, not real Torch/math',phases=20,warm=4,formal=16,queries=160,committed_appends=140,commands=1190,expected_fences=1270,partial_failure_saved_closed=True)))
if __name__=='__main__':main()
