from pathlib import Path
import ast,datetime,hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-sdxl-backward-audit-CPU-v1';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];transport=json.loads((D/'transport-outcome.json').read_bytes());request=transport['request_id'];assert transport['transport_rc']==0 and transport['remote_exec_acknowledged']
task=json.loads(subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':coordination/tasks/training-sdxl-backward-audit-CPU-v1.json']));pins={name:task['source_SHA256'][name] for name in task['remote_files']}
REMOTE=r'''
from pathlib import Path
import hashlib,json,socket,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');W=R/'results/training-sdxl-backward-audit-CPU-v1';evidence={}
def read(n,cap=2*1024**2):
 p=W/n;assert p.is_file() and not p.is_symlink() and p.stat().st_size<=cap;raw=p.read_bytes();evidence[n]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return raw
intent=json.loads(read('intent.json'));outcome=json.loads(read('outcome.json'));result=json.loads(read('evidence/CPU-result.json'));log=read('evidence/tests.private.txt').decode();read('CPU.log');owner=json.loads(read('owned-srun.json'))
assert intent['claim']==outcome['claim']==CLAIM and intent['request_id']==outcome['request_id']==REQUEST and outcome['deadline']==intent['deadline']==intent['epoch']+300 and result['epoch']==intent['epoch'] and result['deadline']==intent['epoch']+220
assert intent['node']=='f11r4n11' and intent['parent']=='1087656' and len(intent['actual_nodes'])==len(set(intent['actual_nodes']))==40
for name,pin in PINS.items():assert hashlib.sha256(read('source/'+name)).hexdigest()==pin and intent['source_files'][name]==pin
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];lines=subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines();rows=[row.split('|') for row in lines if '|training-sdxl-backward-audit-CPU-v1|' in row];assert len(rows)==1 and rows==outcome['accounting'] and outcome['accounting_columns']==columns and rows[0][2:4]==['COMPLETED','0:0'] and rows[0][5]=='f11r4n11' and rows[0][0]==result['owner']['step']
assert outcome['passed'] and outcome['returncode']==0 and outcome['owned_srun_reaped'] and not outcome['forced_cleanup'] and outcome['CPU-result.json']['SHA256']==evidence['evidence/CPU-result.json']['SHA256']
assert result['passed'] and result['tests']==9 and not result['CUDA_initialized'] and result['GPU_cards']==result['model_weight_bytes_read']==result['raw_input_bytes_read']==0 and result['actual_source_subset_files']==5 and result['total_framework_files']==330 and not result['full330_framework_installed'] and not result['full_UNet_instantiated'] and not result['native_model_backward_observed'] and not result['numerical_alignment'] and not result['Model_GO']
assert 'Ran 9 tests' in log and '\nOK\n' in log and log.count(' ... ok')==9 and hashlib.sha256(log.encode()).hexdigest()==result['tests_log_SHA256']
assert result['framework_commit']==FRAMEWORK and result['framework_tree']==TREE and result['versions']==json.loads(read('source/DEPENDENCIES.json'))['versions']
same_host=owner['host']==socket.gethostname();original_srun_alive=None
if same_host:
 p=Path('/proc')/str(owner['PID'])/'stat';original_srun_alive=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==owner['startticks'];assert not original_srun_alive
print(json.dumps(dict(independent_passed=True,claim=CLAIM,request_id=REQUEST,epoch=intent['epoch'],deadline=intent['deadline'],accounting_columns=columns,actual_accounting=rows,original_srun_same_host_observable=same_host,original_srun_alive=original_srun_alive,CPU_result=result,evidence=evidence,source_files_verified=len(PINS),real_UNet_backward_qualified=False,Model_GO=False)))
'''
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text();ns={'__file__':str(D/'reader_reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nPINS='+repr(pins)+'\nFRAMEWORK='+repr(task['framework_commit'])+'\nTREE='+repr(task['framework_tree'])+'\n').encode()+REMOTE.encode()
with (D/'independent-reader-intent.json').open('x') as stream:json.dump(dict(claim=claim,request_id=request,source_SHA256=hashlib.sha256(payload).hexdigest(),reader_source_file_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),readonly=True,new_Slurm=0,new_GPU=0),stream,indent=2)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=35);rc=r.returncode;stdout=r.stdout;stderr=r.stderr
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b''
for name,raw in [('independent-reader.stdout.private.json',stdout),('independent-reader.stderr.private.txt',stderr)]:
 with (D/name).open('xb') as stream:stream.write(raw)
value=json.loads(stdout) if rc==0 else None
record=dict(returncode=rc,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_request=request,claim=claim,stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),value=value,readonly=True,GPU_cards=0)
with (D/'independent-original-terminal.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(dict(returncode=rc,independent_passed=value.get('independent_passed') if value else None,actual_accounting=value.get('actual_accounting') if value else None,tests=value.get('CPU_result',{}).get('tests') if value else None,source_files_verified=value.get('source_files_verified') if value else None,Model_GO=False)))
