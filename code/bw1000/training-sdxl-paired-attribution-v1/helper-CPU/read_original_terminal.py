from pathlib import Path
import hashlib,json,socket,subprocess,time
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-sdxl-paired-attribution-helper-CPU-v1');started=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
intent=json.loads((W/'intent.json').read_bytes());out=json.loads((W/'outcome.json').read_bytes());owned=json.loads((W/'owned-srun.json').read_bytes());q=json.loads((W/'evidence/CPU-result.json').read_bytes());assert intent['claim']==out['claim']==CLAIM and intent['request_id']==out['request_id']==REQUEST and intent['epoch']==out['epoch']==q['epoch'] and intent['deadline']==out['deadline']==intent['epoch']+300 and q['deadline']==intent['epoch']+220
assert out['passed'] and out['returncode']==0 and not out['forced_cleanup'] and out['owned_srun_reaped'];assert out['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'] and len(out['accounting'])==1;row=out['accounting'][0];assert row[1]=='training-sdxl-paired-attribution-helper-CPU-v1' and row[2:4]==['COMPLETED','0:0'] and row[5]=='f11r4n11' and q['owner']['step']==row[0]
now=[s.split('|') for s in subprocess.check_output(['sacct','-n','-P','-j',row[0],'--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if s.split('|')[0]==row[0]];assert now==[row]
pins={}
for n in REMOTE_FILES:
 p=W/'source'/n;assert p.is_file() and not p.is_symlink() and sha(p)==SOURCES[n]==intent['source_files'][n];pins[n]=sha(p)
assert len(pins)==15;assert sha(W/'CPU.log')==out['log_SHA256'];assert sha(W/'evidence/CPU-result.json')==out['CPU-result.json']['SHA256'] and q==out['CPU-result.json']['value'];assert q['passed'] and q['tests']==12 and q['legacy_rank2_FP32_tests']==3 and q['rank4_BF16_tests']==5 and q['entry_guard_tests']==4 and not q['CUDA_initialized'] and not q['real_operand_archives_read'] and not q['paired_attribution_executed'] and not q['Model_GO'] and q['GPU_cards']==0
log=(W/'evidence/tests.private.txt').read_text();assert 'Ran 12 tests' in log and '\nOK\n' in log and 'FAILED' not in log
maps={}
for n,pin in q['providers'].items():
 native=Path(pin['path']);assert native.is_absolute()
 if str(native).startswith('/training-runtime/'):p=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-t06-sdxl-runtime-v1')/native.relative_to('/training-runtime')
 else:assert str(native).startswith('/usr/local/');p=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1')/str(native).lstrip('/')
 assert p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];maps[n]=dict(container_path=str(native),HOST_path=str(p),SHA256=sha(p))
alive=None
if owned['host']==socket.gethostname():
 try:s=Path('/proc',str(owned['PID']),'stat').read_text().split(') ',1)[1].split();alive=int(s[19])==owned['startticks']
 except FileNotFoundError:alive=False
 assert alive is False
answer=dict(evidence_verified=True,original_outcome=out,actual_CPU_result=q,fresh_accounting=now,accounting_columns=out['accounting_columns'],source_files_verified=pins,provider_maps=maps,original_owner_alive=alive,reader_host=socket.gethostname(),artifacts={n:dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n)) for n in ['intent.json','outcome.json','owned-srun.json','CPU.log','evidence/CPU-result.json','evidence/tests.private.txt']},reader_imported_Torch=False,reader_raw_operand_archives_read=False,reader_wall_seconds=time.monotonic()-started);print(json.dumps(answer))
