"""Reference an immutable post-destroy report without enlarging control frames."""
from pathlib import Path
import hashlib,json,os,re,time

def name(run,role):
 assert re.fullmatch('[a-z0-9-]+',run)and role in ['P','D']
 return 'Deep382-'+run+'-'+role+'-shutdown.json'

def summary(report):
 return dict(run=report['run'],role=report['role'],actual_step=report['actual_step'],source_manifest_SHA256=report['source_manifest_SHA256'],
  stage_report_count=len(report['stage_reports']),actual_Native=report['actual_Native'],
  normal_actual8owner_destroy=report['normal_actual8owner_destroy'],normal_stage_loop_close=report['normal_stage_loop_close'])

def publish(root,run,role,report):
 assert report['run']==run and report['role']==role
 value=summary(report)
 assert value['actual_Native']and value['normal_actual8owner_destroy']and value['normal_stage_loop_close']
 assert 'error'not in report and not report['completed']
 path=Path(root)/'results'/name(run,role)
 raw=(json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf8')
 # This file is never rewritten by the role's subsequent socket-close finally.
 with path.open('xb')as f:
  f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(kind='stopped',role=role,report_ref=dict(path='/refactor/results/'+path.name,SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),summary=value)

def consume(root,run,role,frame,deadline):
 if time.monotonic()>=deadline:raise TimeoutError('Original cleanup deadline expired')
 assert set(frame)=={'kind','role','report_ref','summary'}and frame['kind']=='stopped'and frame['role']==role
 ref=frame['report_ref'];assert set(ref)=={'path','SHA256','bytes'}
 assert ref['path']=='/refactor/results/'+name(run,role)
 assert type(ref['bytes'])is int and 0<ref['bytes']<=32<<20
 assert re.fullmatch('[0-9a-f]{64}',ref['SHA256'])
 path=Path(root)/'results'/name(run,role)
 until=min(deadline,time.monotonic()+90)
 while True:
  try:raw=path.read_bytes()
  except FileNotFoundError:raw=None
  if raw is not None and len(raw)>=ref['bytes']:break
  # Only passive visibility/short publication waits. A complete wrong hash,
  # oversized file or mismatched identity fails immediately below.
  if time.monotonic()>=until:raise TimeoutError('Original cleanup deadline/immutable report visibility bound')
  time.sleep(min(.02,max(0,until-time.monotonic())))
 assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['SHA256']
 report=json.loads(raw);assert report['run']==run and report['role']==role
 expected=summary(report)
 assert frame['summary']==expected and all(expected[k]is True for k in ['actual_Native','normal_actual8owner_destroy','normal_stage_loop_close'])
 assert type(expected['stage_report_count'])is int and not report['completed']and'error'not in report
 return report
