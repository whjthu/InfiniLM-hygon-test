"""Bounded NFS visibility only, never repeat a model request or residency."""
import json,pathlib,time
def terminal_report(path,visibility_s=90):
 end=time.monotonic()+visibility_s
 while True:
  try:q=json.loads(pathlib.Path(path).read_bytes())
  except(FileNotFoundError,json.JSONDecodeError):q=None
  if isinstance(q,dict)and q.get('completed')is True:return q
  if time.monotonic()>=end:raise TimeoutError('Terminal artifact not visible/completed: '+str(path))
  time.sleep(1)
def terminal_log(path,normal_marker,visibility_s=90):
 end=time.monotonic()+visibility_s
 while True:
  try:text=pathlib.Path(path).read_text(errors='replace')
  except FileNotFoundError:text=''
  if normal_marker in text:return text
  if time.monotonic()>=end:raise TimeoutError('Terminal log/normal marker not visible: '+str(path))
  time.sleep(1)
