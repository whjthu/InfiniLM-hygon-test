"""CPU report publication only; never hold this lock over a native call/barrier.
Each owner keeps its own mutable record. Shared state owns deep, isolated copies.
"""
import copy,json,pathlib,threading
class OwnerSnapshotReport:
 def __init__(self,path,initial,*,encode=None):
  self._path=pathlib.Path(path);self._lock=threading.RLock();self._state=copy.deepcopy(initial);self._encode=encode or(lambda x:json.dumps(x,indent=2))
 def _write_locked(self):
  encoded=self._encode(self._state);tmp=self._path.with_suffix('.tmp');tmp.write_text(encoded,encoding='utf8');tmp.replace(self._path)
 def publish_worker(self,rank,record):
  # record belongs exclusively to this calling worker. No reference is retained.
  if type(rank)is not int or record.get('rank')!=rank:raise ValueError('rank ownership differs')
  with self._lock:
   if not 0<=rank<len(self._state['workers']):raise ValueError('rank outside report')
   self._state['workers'][rank]=copy.deepcopy(record);self._write_locked()
 def publish_top(self,**fields):
  if 'workers'in fields:raise ValueError('workers must use isolated owner publication')
  with self._lock:self._state.update(copy.deepcopy(fields));self._write_locked()
 def flush(self):
  with self._lock:self._write_locked()
 def snapshot(self):
  with self._lock:return copy.deepcopy(self._state)
