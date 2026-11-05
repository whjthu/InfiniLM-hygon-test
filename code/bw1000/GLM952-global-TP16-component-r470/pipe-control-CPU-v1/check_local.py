"""Small local actual-framing consumer; actor records are explicit recording."""
import concurrent.futures,importlib.util,json,socket,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('r470_control',Path(__file__).with_name('control.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 ident=dict(label='recording-only',step='none');records=[[dict(identity=ident,phase='created',rank=n*8+d,node=m.NODES[n],device=d,PID=100+n*8+d,startticks='1')for d in range(8)]for n in range(2)];a,b=socket.socketpair()
 with a,b,concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
  jobs=[pool.submit(m.exchange,s,n,records[n],'created',ident,time.time())for n,s in enumerate([a,b])];digests=[f.result(timeout=3)for f in jobs];assert len(set(digests))==1
 negatives=[]
 for label,change in [('rank',lambda r:r[0].update(rank=8)),('identity',lambda r:r[0].update(identity={})),('phase',lambda r:r[0].update(phase='proved'))]:
  r=json.loads(json.dumps(records[0]));change(r)
  try:m.validate(r,0,'created',ident)
  except AssertionError:negatives.append(label)
  else:raise AssertionError('Invalid '+label+' accepted')
 # Actual receive on a closed peer, bounded; no protocol reconnection.
 a,b=socket.socketpair();b.close()
 try:
  try:m.recv(a,time.time())
  except EOFError:negatives.append('EOF')
  else:raise AssertionError('EOF accepted')
 finally:a.close()
 try:m.limit(time.time()-151)
 except TimeoutError:negatives.append('original-deadline')
 else:raise AssertionError('Expired deadline accepted')
 print(json.dumps(dict(local_CPU_passed=True,actual_framing_exchange=True,recording_actor_records=True,negative_rejected=negatives,SDK=False,Runtime=False,GPU=False,Slurm=False)))
if __name__=='__main__':main()
