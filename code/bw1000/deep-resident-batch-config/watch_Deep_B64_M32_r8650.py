import pathlib,json,subprocess,time,datetime,sys
from deep_causal_observation_r7233 import terminal_state,observed_run
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/deep-B64-M32-r8650-watcher.json';intent=r/'results/deep-B64-M32-r8650-watcher-intent.json';statepath=r/'results/deep-B64-M32-r8650-watcher-state.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(record):
 record['utc']=now();tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(statepath)
if '--resume'in sys.argv:
 assert intent.exists()and statepath.exists()and not out.exists(),'Resume only UNKNOWN existing watcher; never resubmit numerical job'
 record=json.loads(statepath.read_text());assert record.get('resumable_samehandle_only')
else:
 assert not out.exists()and not statepath.exists()
 with intent.open('x')as f:json.dump({'utc':now(),'no_resubmit':True},f)
 record={'step':None,'terminal':None,'no_resubmit':True,'performance_adopted':False,'observations':[],'resumable_samehandle_only':True}
began=time.monotonic();save(record)
while not record['terminal']and time.monotonic()-began<2600:
 p=observed_run(['sacct','-S','2026-10-24','-j','910374','-n','-P','-o','JobID,JobName%60,State%40,ExitCode,Elapsed,NodeList%100'],90)
 if not p['observed']or p.get('exit')!=0:
  record['status']='UNKNOWN';record['observations'].append({'utc':now(),'sacct':p});save(record);time.sleep(30);continue
 matches=[v.split('|')for v in p['stdout'].splitlines()if len(v.split('|'))>=6 and v.split('|')[1]=='Deep-B64-M32-r8650']
 if len(matches)>1:record['status']='UNKNOWN_MULTIPLE_MATCHES';record['observations'].append({'utc':now(),'matches':matches});save(record);break
 if matches:
  row=matches[0]
  if record['step']not in[None,row[0]]:record['status']='UNKNOWN_HANDLE_CHANGED';record['observations'].append({'utc':now(),'unexpected':row});save(record);break
  record['step']=row[0];record['last_row']=row;record['status']='OBSERVING_SAME_HANDLE'
  if terminal_state(row[2]):record['terminal']=row
 else:record['status']='UNKNOWN_HANDLE_NOT_YET_OBSERVED'
 save(record)
 if not record['terminal']:time.sleep(30)
if not record['terminal']:
 record['status']='UNKNOWN_OBSERVATION_ENDED';record['observation_timeout_only_no_restart']=True;save(record);print(json.dumps(record));sys.exit(0)
if 'audit'not in record:
 if record['terminal'][2:4]==['COMPLETED','0:0']:
  record['audit']=observed_run(['python3',str(r/'incoming/audit_Deep_B64_M32_r8650.py'),record['step']],180)
 else:record['audit']={'skipped':'numerical_step_not_COMPLETED0'}
 save(record)
# Any authoritative terminal requires cleanup observation, irrespective of audit outcome.
post=observed_run(['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n04','--gres=none','--mem=512M','-t','00:03:00','--job-name=Deep-B64-M32-postidle-r8650','python3',str(r/'incoming/probe_Deep_idle_r4847.py')],165)
record.setdefault('postKFD_attempts',[]).append({'utc':now(),**post})
if not post['observed']or post.get('exit')!=0:
 record['status']='UNKNOWN_POSTKFD';save(record);print(json.dumps(record));sys.exit(0)
try:
 parsed=json.loads(next(v for v in post['stdout'].splitlines()if v.startswith('{')));assert parsed['passed']and parsed['node']=='a02r4n04'
except (ValueError,StopIteration,KeyError,AssertionError)as e:
 record['status']='UNKNOWN_POSTKFD_CONTENT';record['postKFD_parse_error']=repr(e);save(record);print(json.dumps(record));sys.exit(0)
record.update(status='TERMINAL_POSTKFD_VERIFIED',postKFD=parsed,resumable_samehandle_only=False);save(record)
with out.open('x')as f:json.dump(record,f,indent=2)
print(json.dumps(record))
