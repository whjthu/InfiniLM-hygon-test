import pathlib,json,hashlib,subprocess,os,time,datetime,io,contextlib,runpy
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))}
EXPECTED_RECOVERY='99813434edacf1d487c6f5097a416698cfa6e154e20eaaef1fb192406040bd70'
record=R/'results/Kimi144d-recovered-profile-handoff-r9421.json';assert not record.exists()
p=R/'results/Kimi144d-recoverypost-r9420.json';assert sha(p)==EXPECTED_RECOVERY;probe=json.loads(p.read_bytes());assert probe['all4_KFD_empty']and len(probe['rows'])==4 and {x['step']for x in probe['rows']}=={'910380.442'}and all(not x['kfd_pids']for x in probe['rows'])
age=datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(probe['rows'][0]['utc']);assert 0<=age.total_seconds()<1200
cp=R/'results/Kimi-agent-P0-client-r144d-kimi-agent-p0-r144d-a91b0c6e/result.json';assert sha(cp)=='406a2d2eaf5d3679c55b78e0dedcd261ea6e8e889e2ddb4bc3b9dda47910561f';client=json.loads(cp.read_bytes());assert client['passed']and client['completed']and client['identity_SHA256']=='bbc5030daee4a1400e639a2b73be776dcc00c50d3ee90406ad046bb29ddf34c0'
for i in range(4):
 p=R/'results'/('Kimi-agent-service-r144d-kimi-agent-p0-r144d-a91b0c6e-node%d.json'%i);n=json.loads(p.read_bytes());assert n['node']==i and n['completed']and n['all8_owned_workers_shutdown_completed']and n['protocol_before_destroy_passed']and n['agent_HTTP_normal_shutdown']and not n.get('shutdown_error')
def acct(parent):
 s=subprocess.check_output(['sacct','-S','2026-10-27','-j',parent,'-n','-P','-o','JobID,JobName%120,State,ExitCode'],env=env,text=True,timeout=30);return {x.split('|')[0]:x.split('|')for x in s.splitlines()}
def norm(x):return x.split()[0].rstrip('+')
for _ in range(15):
 a=acct('910380');row=a.get('910380.442')
 if row and norm(row[2])=='COMPLETED':break
 time.sleep(2)
assert row[2:4]==['COMPLETED','0:0']and a['910380.438'][2:4]==['COMPLETED','0:0']and a['910380.439'][2:4]==['COMPLETED','0:0']
wp=R/'results/Kimi-profile-after-P0-r149.json';w=json.loads(wp.read_bytes());assert w['CPU_actual_step']=='910407.483'and w['profile_invoked']is False
assert not(R/'results/Kimi-profile-after-P0-r149-intent.json').exists()
for n in ['Kimi-sparse-profile-r136b-intent.json','Kimi-sparse-profile-r136b-launch.json']:assert not(R/'results'/n).exists()
row=acct('910407')['910407.483'];assert row[1]=='Kimi-profile-after-P0-r149'
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_failed_steps_preserved':['910380.440','910380.441','910407.477'],'recovery_step':'910380.442','recovery_SHA256':EXPECTED_RECOVERY,'real_P0_client_passed':True,'exact_CPU_waiter':'910407.483','profile_invoked':False,'no_P0_restart':True,'no_old_watch_modified':True}
with record.open('x')as f:json.dump(q,f,indent=2)
if norm(row[2])in ['RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED']:subprocess.run(['scancel','--signal=TERM','910407.483'],env=env,check=True,timeout=30)
for _ in range(30):
 row=acct('910407')['910407.483']
 if norm(row[2])not in ['RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED']:break
 time.sleep(2)
assert norm(row[2])in ['CANCELLED','COMPLETED','FAILED','TIMEOUT'];q['waiter_terminal']=row;record.write_text(json.dumps(q,indent=2))
assert json.loads(wp.read_bytes())['profile_invoked']is False
for n in ['Kimi-sparse-profile-r136b-intent.json','Kimi-sparse-profile-r136b-launch.json']:assert not(R/'results'/n).exists()
entry=R/'incoming/Kimi-sparse-profile-Model-r136b/launcher/preflight_then_run.py';assert sha(entry)=='6a7485910ea3626925420815d27bb43136e08cb51a2bdd8f211effea1aaa9486';buf=io.StringIO()
with contextlib.redirect_stdout(buf):runpy.run_path(str(entry),run_name='__main__')
q['profile_invoker_receipt']=json.loads(buf.getvalue());q['profile_invoked']=True;q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
