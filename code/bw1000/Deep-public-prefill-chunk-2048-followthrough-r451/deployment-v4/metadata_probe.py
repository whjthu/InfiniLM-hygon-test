"""One login-HOST metadata child,90s; observe only its process tree, never Model."""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent
OUT=R/'results/Deep451-deployment-v4-metadata'/('node-'+os.uname().nodename+'-pid'+str(os.getpid()))
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,q):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2,allow_nan=False)+'\n');t.replace(p)
def proc(pid):
 p=Path('/proc')/str(pid);q=dict(PID=pid)
 for name in ['stat','cmdline','wchan']:
  try:
   raw=(p/name).read_bytes();q[name]=raw.decode(errors='replace').replace('\x00',' ')
  except OSError as e:q[name+'_error']=repr(e)
 for name in ['cwd','exe']:
  try:q[name]=os.readlink(p/name)
  except OSError as e:q[name+'_error']=repr(e)
 try:q['fdlinks']={x.name:os.readlink(x)for x in(p/'fd').iterdir()}
 except OSError as e:q['fd_error']=repr(e)
 return q
def children(pid):
 try:return[int(x)for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
 except OSError:return[]
def snapshot(pid):
 pending=[pid];seen=set();rows=[]
 while pending:
  value=pending.pop()
  if value in seen:continue
  seen.add(value);row=proc(value);rows.append(row);pending.extend(children(value))
 return dict(monotonic=time.monotonic(),members=rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);p.add_argument('--publication',required=True);a=p.parse_args()
 assert H(D/'SOURCE_MANIFEST.json')==a.source_sha
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
 claim=json.loads((D/'METADATA_RESOURCE_CLAIM.json').read_bytes());assert claim['child_wall_s']==90 and claim['total_wall_s']==300 and claim['GPU']==0
 original=json.loads((D/'ORIGINAL_COMMAND.json').read_bytes());argv=original['argv'];i=argv.index('python3')
 assert argv[i:i+4]==['python3','-B','-m','public_deep_model']and argv[i+4:]==original['CLI_args']and '--serve'not in argv
 # Exactly original container/binds/env plus explicit marker bootstrap and importtime.
 marked=argv[:i]+['python3','-B','-u','-X','importtime','/refactor/incoming/Deep-public-prefill-chunk-2048-followthrough-r451/deployment-v4/bootstrap.py',*original['CLI_args']]
 env={k:v for k,v in os.environ.items()if k not in ['BASH_ENV','ENV','LD_PRELOAD','PYTHONSTARTUP','PYTHONINSPECT']and not k.startswith(('APPTAINER','SINGULARITY'))}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=False);start=time.monotonic();path=OUT/'result.json'
 q=dict(completed=False,publication=a.publication,source_manifest_SHA256=a.source_sha,original_argv=argv,marked_argv=marked,original_c850_SHA256=original['source_manifest_SHA256'],HOST_PID=os.getpid(),HOST_node=os.uname().nodename,original_start_unix=time.time(),child_wall_s=90,total_wall_s=300,process_tree_snapshots=[],signals=[],Native_dlopen=False,Runtime=False,GPU=False,Model_READY=False,Slurm_submitted=False,output_limit_each_bytes=4*1024**2,original_process_or_suite_replayed=False)
 save(path,q);child=None;identity=None
 def own_group():
  if child is None or child.poll()is not None:return False
  try:
   stat=Path(f'/proc/{child.pid}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
   return os.getpgid(child.pid)==child.pid and identity is not None and fields[19]==identity
  except(OSError,IndexError):return False
 def signal_group(kind):
  if not own_group():
   q['cleanup_identity_unknown']=True;save(path,q);return False
  os.killpg(child.pid,getattr(signal,'SIG'+kind));q['signals'].append(dict(signal=kind,known_PID=child.pid,owned_PGID=child.pid,monotonic=time.monotonic()));save(path,q);return True
 def close_known(reason):
  q['bounded_stop_reason']=reason;save(path,q)
  if signal_group('TERM'):
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:
    if signal_group('KILL'):
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:q['owned_child_unclosed']=True
 try:
  with(OUT/'stdout').open('xb')as stdout,(OUT/'stderr').open('xb')as stderr:
   child=subprocess.Popen(marked,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
   created=time.monotonic();q.update(child_PID=child.pid,owned_PGID=child.pid,child_start_monotonic=created)
   try:
    stat=Path(f'/proc/{child.pid}/stat').read_text();identity=stat[stat.rfind(')')+2:].split()[19];q['PID_start_ticks']=identity;q['actual_PGID']=os.getpgid(child.pid);assert q['actual_PGID']==child.pid
   except(FileNotFoundError,ProcessLookupError):q['exited_before_identity_capture']=True
   save(path,q);seen=set()
   while child.poll()is None:
    elapsed=time.monotonic()-created
    sizes={n:(OUT/n).stat().st_size for n in ['stdout','stderr']}
    if any(v>4*1024**2 for v in sizes.values()):
     q['output_limit_exceeded_observed_bytes']=sizes;close_known('parent_observed_output_limit');break
    for t in [1,85]:
     if elapsed>=t and t not in seen:
      seen.add(t);q['process_tree_snapshots'].append(dict(at_s=t,observation=snapshot(child.pid)));save(path,q)
    if elapsed>=90:
     q['timed_out_original90s']=True;save(path,q)
     close_known('original90s_child_deadline')
     break
    time.sleep(min(.1,max(0,90-elapsed)))
   q['returncode']=child.poll();q['child_wall_elapsed_s']=time.monotonic()-created
 except BaseException as error:
  q.update(error=repr(error),traceback=traceback.format_exc())
  if child is not None and child.poll()is None:close_known('parent_exception')
 finally:
  records=[]
  for name in ['stdout','stderr']:
   p=OUT/name
   if p.exists():
    actual_bytes=p.stat().st_size
    with p.open('rb')as f:raw=f.read(4*1024**2)
    truncated=actual_bytes>len(raw)
    if truncated and child is not None and child.poll()is not None:
     with p.open('r+b')as f:f.truncate(len(raw))
    q[name]=dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),observed_original_bytes=actual_bytes,truncated=truncated,hash_scope='retained_prefix'if truncated else'complete_file',tail_omitted=truncated)
    if name=='stderr':
     q['Python_ENTER_observed']=b'DEEP451_PYTHON_ENTER\n'in raw
     for line in raw.decode(errors='replace').splitlines():
      if line.startswith('DEEP451_DIAG '):
       try:records.append(json.loads(line[len('DEEP451_DIAG '):]))
       except json.JSONDecodeError:pass
     q['importtime_lines']=sum(x.startswith('import time:')for x in raw.decode(errors='replace').splitlines())
  q['last_marker']=records[-1]if records else None;q['last_OPEN_ATTEMPT']=next((x for x in reversed(records)if x['kind']=='OPEN_ATTEMPT'),None);q['marker_count']=len(records)
  q['actual_original_CLI_returned']=any(x['kind']=='ORIGINAL_MODULE_RETURNED'for x in records)
  if child is not None:
   try:os.killpg(child.pid,0);q['owned_PGID_still_live']=True
   except ProcessLookupError:q['owned_PGID_still_live']=False
   except PermissionError:q['owned_PGID_still_live']='UNKNOWN'
  if q.get('returncode')==0 and q['actual_original_CLI_returned']:
   try:q['actual_CLI_result']=json.loads((OUT/'stdout').read_bytes());q['actual_schema_valid']=q['actual_CLI_result']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False)
   except ValueError:q['actual_schema_valid']=False
  q.update(completed=True,total_wall_s_actual=time.monotonic()-start,root_cause='UNKNOWN; phase markers only, one return does not explain original timeout',no_Model_or_GPU_authorization=True)
  save(path,q)
 print(json.dumps(dict(output=str(path),completed=q['completed'],returncode=q.get('returncode'),Python_ENTER=q.get('Python_ENTER_observed'),last_marker=q.get('last_marker'),actual_schema_valid=q.get('actual_schema_valid'),timed_out=q.get('timed_out_original90s',False),GPU=False)))
if __name__=='__main__':main()
