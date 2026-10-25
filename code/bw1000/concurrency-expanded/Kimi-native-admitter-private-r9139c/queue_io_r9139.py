import json,os

def canonical(cmd):return json.dumps(cmd,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()+b'\n'
def publish(path,cmd):
 raw=canonical(cmd);temp=path.parent/('.admit-'+os.urandom(8).hex()+'.tmp')
 with temp.open('wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 try:
  os.link(temp,path);return True
 except FileExistsError:assert json.loads(path.read_text())==cmd,'conflictingimmutablequeuefile';return False
 finally:temp.unlink()
