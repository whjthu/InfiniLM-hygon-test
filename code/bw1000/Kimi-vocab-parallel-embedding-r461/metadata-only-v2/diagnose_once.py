"""Independent metadata-only source proposal. No launcher/keeper/Slurm path.

Never replay device-v1. Execution requires ordinary source publication first.
Uses real stdout/stderr files, with bounded owned process cleanup. No pipes.
"""
from pathlib import Path
import argparse,hashlib,json,os,signal,subprocess,sys,time
D=Path(__file__).resolve().parent
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
P=R/'incoming/Kimi-vocab-parallel-embedding-r461/device-v1'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--manifest-sha',required=True);a=parser.parse_args(argv)
 assert H(D/'SOURCE_MANIFEST.json')==a.manifest_sha
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert m['metadata_only']and not m['GPU_GO']and not m['Model_GO']
 assert m['HOST_execution_authorized'],'Source-only draft: publish exact authorized metadata scope before execution'
 for name,h in m['files_SHA256'].items():assert H(D/name)==h,name
 assert H(P/'SOURCE_MANIFEST.json')=='932cafea926f90f5c644281724e166b6965ac961818952c34001dfed7f769473'
 for name,h in json.loads((P/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert H(P/name)==h,name
 sys.path.insert(0,str(P));import controller as parent
 out=R/'results/Kimi461-metadata-only-v2-original';assert not out.exists();out.mkdir()
 result=dict(metadata_only=True,GPU=0,Slurm=0,active_candidate_Native_calls=0,explicit_SDK_calls=0,
  original_preload_mapping_or_initialization='UNKNOWN; original LD_PRELOAD/environment retained as diagnostic condition',
  source_manifest_SHA256=a.manifest_sha,started_unix=time.time(),overall_limit_s=90,child_work_limit_s=80,
  phase='verified_source_before_Popen',original_device_v1_replayed=False)
 started=time.monotonic();stop=started+90
 def save():
  t=out/'result.tmp';t.write_text(json.dumps(result,indent=2)+'\n');t.replace(out/'result.json')
 save()
 env=parent.clean_environment(os.environ);env.update(K461_METADATA_ONLY='1',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',SLURM_JOB_ID='recording-HOST',SLURM_STEP_ID='metadata-only',SLURM_PROCID='0')
 cmd=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--home','/public/home/aclsylqidf/hygon-inference-20261013/build-deps/root:/root',
  '--bind',str(R)+':/refactor','--bind','/public/home/aclsylqidf/shared:/shared:ro',
  '--bind','/public/home/aclsylqidf/hygon-inference-20261013:/old:ro','--bind','/dev/shm:/local',
  str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc',str(Path('/refactor')/D.relative_to(R)/'metadata_inside.sh')]
 result['argv']=cmd
 # Files exist before Popen and remain available even if a descendant retains
 # its stdio descriptors. proc.wait does not await descendant pipe EOF.
 with(out/'stdout.txt').open('xb')as stdout,(out/'stderr.txt').open('xb')as stderr:
  proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  result.update(phase='metadata_child_running',child_PID=proc.pid,owned_PGID=proc.pid,UID=os.getuid());save()
  try:proc.wait(timeout=max(1,min(80,stop-time.monotonic()-10)));result['timeout']=False
  except subprocess.TimeoutExpired:
   result['timeout']=True
  def group_state():
   proc.poll() # reap the parent if it exited; never equate that with group exit
   try:os.killpg(proc.pid,0);return True
   except ProcessLookupError:return False
   except OSError as error:
    result.setdefault('group_observation_errors',[]).append(repr(error));return None
  group_live=group_state()
  # This also runs after NORMAL parent exit: a same-PGID descendant must not
  # be left alive merely because proc.wait returned successfully.
  if group_live is not False:
   for kind in [signal.SIGTERM,signal.SIGKILL]:
    if time.monotonic()>=stop:break
    try:
     os.killpg(proc.pid,kind)
     result.setdefault('signals',[]).append(dict(PGID=proc.pid,signal=kind.name))
    except ProcessLookupError:
     group_live=False;break
    except OSError as error:
     result.setdefault('signal_errors',[]).append(dict(PGID=proc.pid,signal=kind.name,error=repr(error)))
    save()
    until=min(stop,time.monotonic()+5)
    while time.monotonic()<until:
     group_live=group_state()
     if group_live is False:break
     time.sleep(min(.05,max(0,until-time.monotonic())))
    group_live=group_state()
    if group_live is False:break
  group_live=group_state();parent_rc=proc.poll()
  terminal_unknown=parent_rc is None or group_live is not False
  result.update(returncode=parent_rc,parent_rc=parent_rc,own_group_still_live=group_live,group_gone=group_live is False,
   terminal_UNKNOWN=terminal_unknown,elapsed_s=time.monotonic()-started,
   phase='metadata_child_terminal_UNKNOWN'if terminal_unknown else'metadata_child_and_group_terminal')
 save()
 # Only small file metadata is emitted. The trace/source/raw streams stay R.
 result['stdio_files']={name:dict(path=str(out/name),bytes=(out/name).stat().st_size,SHA256=H(out/name))for name in ['stdout.txt','stderr.txt']}
 result['passed']=result['parent_rc']==0 and not result.get('timeout')and result['group_gone']and not result['terminal_UNKNOWN']
 save();print(json.dumps(result))
if __name__=='__main__':main()
