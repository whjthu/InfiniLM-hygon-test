"""Byte-preserved stdlib wire/history/listener helpers from public425."""
import hashlib,json,os,time
from pathlib import Path
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,q):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(path)
def opaque_history(body):
 calls={};pairs=[]
 for m in body['messages']:
  if m['role']=='assistant':
   for call in m.get('tool_calls',[]):
    assert call['id']and call['id']not in calls;calls[call['id']]=call
  elif m['role']=='tool':
   key=m['tool_call_id'];assert key in calls
   pairs.append(dict(id=key,function=calls[key]['function']['name'],result_SHA256=sha(m['content'].encode())))
 return pairs
def wait_json(path,deadline,predicate=lambda q:True):
 first=None
 while time.time()<deadline:
  try:
   q=json.loads(Path(path).read_bytes())
   if predicate(q):return q
  except(FileNotFoundError,json.JSONDecodeError):pass
  first=time.monotonic()if first is None else first
  if time.monotonic()-first>90:raise TimeoutError('Bounded artifact visibility:'+str(path))
  time.sleep(.1)
 raise TimeoutError('Original service deadline')
def wait_listener(pid,port,deadline):
 while time.time()<deadline:
  inodes=set()
  for p in Path('/proc/'+str(pid)+'/fd').iterdir():
   try:target=os.readlink(p)
   except OSError:continue
   if target.startswith('socket:['):inodes.add(target[8:-1])
  for name in ['tcp','tcp6']:
   for line in Path('/proc/net/'+name).read_text().splitlines()[1:]:
    row=line.split()
    if row[3]=='0A'and int(row[1].split(':')[1],16)==port and row[9]in inodes:return
  time.sleep(.1)
 raise TimeoutError('Public frontend listener absent; no connection retry')

