from pathlib import Path
import base64,hashlib,json,socket,subprocess,time
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hymotion-target-meta-CPU-v1');started=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
intent=json.loads((W/'intent.json').read_bytes());out=json.loads((W/'outcome.json').read_bytes());owned=json.loads((W/'owned-srun.json').read_bytes());assert intent['claim']==out['claim']==CLAIM and intent['request_id']==out['request_id']==REQUEST and intent['epoch']==out['epoch'] and out['deadline']==intent['deadline']==intent['epoch']+300;assert out['owned_srun_reaped'] and not out['forced_cleanup'];assert sha(W/'outcome.json')==EXPECTED_OUTCOME_SHA if EXPECTED_OUTCOME_SHA else True
pins={}
for n in REMOTE_FILES:
 p=W/'source'/n;assert p.is_file() and not p.is_symlink() and sha(p)==SOURCES[n]==intent['source_files'][n];pins[n]=sha(p)
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];assert out['accounting_columns']==columns and len(out['accounting'])==1;row=out['accounting'][0];assert row[1]=='training-hymotion-target-meta-CPU-v1' and row[5]=='f11r4n15';assert row[2] in ['COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED']
fresh=[line.split('|') for line in subprocess.check_output(['sacct','-n','-P','-j',row[0],'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if line.split('|')[0]==row[0]];assert fresh==[row]
p=W/'CPU.log';assert p.stat().st_size<262144 and sha(p)==out['log_SHA256'];raw=p.read_bytes();original_alive=None
if owned['host']==socket.gethostname():
 try:stat=Path('/proc',str(owned['PID']),'stat').read_text().split(') ',1)[1].split();original_alive=int(stat[19])==owned['startticks']
 except FileNotFoundError:original_alive=False
 assert original_alive is False
qpath=W/'evidence/CPU-result.json';result=json.loads(qpath.read_bytes()) if qpath.is_file() else None
if result:
 assert sha(qpath)==out['CPU-result.json']['SHA256'] and result==out['CPU-result.json']['value'];assert result['owner']['step']==row[0]
answer=dict(evidence_verified=True,qualification_passed=bool(out['passed']),original_outcome=out,actual_CPU_result=result,fresh_accounting=fresh,accounting_columns=columns,source_files_verified=pins,original_intent=intent,original_owner=owned,original_owner_alive=original_alive,reader_host=socket.gethostname(),original_artifacts={n:dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n)) for n in ['intent.json','owned-srun.json','outcome.json','CPU.log']},CPU_log_base64=base64.b64encode(raw).decode(),reader_imported_Torch=False,reader_GPU_cards=0,weights_or_cache_bytes_read=0,reader_wall_seconds=time.monotonic()-started)
print(json.dumps(answer,allow_nan=False))
