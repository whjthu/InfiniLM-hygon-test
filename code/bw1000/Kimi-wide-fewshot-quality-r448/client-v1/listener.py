"""Unchanged r434 passive PID-owned LISTEN observation; no health HTTP."""
from pathlib import Path
import os,time
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

