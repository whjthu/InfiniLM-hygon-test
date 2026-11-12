from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path(__file__).parent;D=B/'training-sdxl-backward-cancellation-v1';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');original=json.loads((D/'transport-outcome.json').read_bytes());h=original['value'];claim=original['claim_commit'];request=original['request_id'];task=json.loads(subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':coordination/tasks/training-sdxl-backward-cancellation-v1.json']));pins={n:task['source_SHA256'][n] for n in task['remote_files']}
REMOTE=r'''
from pathlib import Path
import hashlib,json,math,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-sdxl-backward-cancellation-v1');evidence={}
def sha(p):
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
 return digest.hexdigest()
def read(name,cap=2*1024**2):
 p=W/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<=cap;raw=p.read_bytes();evidence[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());return raw
intent=json.loads(read('intent.json'));result=json.loads(read('controller-result.json'));owner=json.loads(read('keeper-owner.json'));assert owner==OWNER and intent['claim_commit']==result['claim_commit']==CLAIM and intent['request_id']==result['request_id']==REQUEST and intent['epoch']==result['epoch']==OWNER['epoch'] and intent['deadline']==result['deadline']==OWNER['deadline']==intent['epoch']+1200
assert result['completed'] and result['actual_GPU_step']=='1087656.176' and result['framework_commit']==FRAMEWORK
for name,pin in PINS.items():assert hashlib.sha256(read('src/'+name,300000)).hexdigest()==pin and intent['files'][name]==pin
fm=json.loads(read('src/FRAMEWORK_SUBSET.json'));assert fm['executed_subset_files']==len(fm['files'])==8 and fm['full_registered_files']==330 and fm['commit']==FRAMEWORK
for name,pin in fm['files'].items():assert hashlib.sha256(read('source/'+name,300000)).hexdigest()==pin['SHA256']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];rows=[line.split('|') for line in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if '|training-sdxl-backward-cancellation-v1' in line];assert len(rows)==len(result['runs'])==4
assert [(x[0],x[1],x[2],x[3],x[5]) for x in rows]==[(r['terminal']['row'][0],r['name'],r['terminal']['row'][2],r['terminal']['row'][3],r['terminal']['row'][5]) for r in result['runs']]
for r in result['runs']:
 assert not r['forced_cleanup'] and r['terminal']['terminal_visible'] and r['accounting_columns']==columns and r['terminal']['row'][5]=='f11r4n11';assert hashlib.sha256(read(r['name']+'.log',4*1024**2)).hexdigest()==r['log_SHA256']
gpu=result['GPU_receipt'];assert gpu['returncode']==0 and gpu['passed'] and gpu['GPU_cards']==1 and gpu['terminal']['row'][2:4]==['COMPLETED','0:0'];post=result['runs'][2];assert post['name'].endswith('-post') and post['passed'] and post['GPU_cards']==0
before=json.loads(read('before.json'));after=json.loads(read('after.json'));lease=json.loads(read('lease-before.json'));assigned=json.loads(read('assigned.json'));assert before['node']==after['node']==lease['node']==assigned['node']=='f11r4n11';assert before['pci']==after['pci']==lease['pci'] and len(before['pci'])==len(set(before['pci']))==8 and before['boot_id']==after['boot_id']==lease['boot_id']==assigned['boot_id'];assert before['kfd_holder_count']==after['kfd_holder_count']==lease['kfd_holder_count']==0
assert assigned['step']==lease['step']=='1087656.176' and assigned['visible_cards']==1 and assigned['pci'] in before['pci'] and assigned['KFD']['gpu_id']>0 and not assigned['KFD']['other_devices_gpu_id_read'] and not assigned['assigned_PCI_pending'];assert assigned['policies_before']==assigned['policies_after'] and assigned['policies_before']['BF16_reduction'] is True
audit=json.loads(read('candidate/backward-audit.json'));independent=json.loads(read('backward-independent.json'));assert audit==result['backward_audit'] and independent==result['independent_backward'] and audit['passed'] and independent['passed'] and not audit['numerical_repair_applied'] and audit['original_update_gate_unchanged'] and audit['cross_host_raw_tensor_transfer_bytes']==0
diagnostic=audit['diagnostic'];assert diagnostic['module']==MODULE and all(x==1 for x in diagnostic['counts'].values()) and diagnostic['local_reference_device']=='cpu' and diagnostic['local_reference_dtype']=='torch.float64' and diagnostic['captured_tensor_bytes']==independent['captured_tensor_bytes']<=16*1024**2 and not independent['GPU_initialized']
for group,values in independent['computed'].items():
 for key,value in values.items():
  if type(value) is float:assert math.isclose(value,diagnostic[group][key],rel_tol=1e-10,abs_tol=1e-15)
  else:assert value==diagnostic[group][key]
assert independent['cancellation_ratio_at_least']==diagnostic['cancellation_ratio_at_least'] and independent['exact_zero_with_nonzero_absolute_sum']==diagnostic['exact_zero_with_nonzero_absolute_sum']
capture=W/'candidate/local-operands.private.pt';assert capture.stat().st_size==audit['private_capture']['bytes']<17*1024**2 and sha(capture)==audit['private_capture']['SHA256']==independent['private_capture_SHA256']
raw=read('comparison.json',4*1024**2);full=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==result['comparison']['comparison_SHA256'] and full['checks']==result['comparison']['checks'] and full['passed']==result['numerical_passed'] and full['reference_sha256']=='c59a71095924edfdfb9ad7041b8ef32f747e7825557a313f72ad09a3b0b53036' and full['policy_sha256']=='075e86e54a3eeefb8d5aad219a23f16959d14bd42b22103cd5fe5302c823f6b5'
for name,metric in full['metrics'].items():
 if 'per_tensor' not in metric:continue
 values=list(metric['per_tensor'].values());a=math.sqrt(math.fsum(x['observed_l2_squared'] for x in values));b=math.sqrt(math.fsum(x['reference_l2_squared'] for x in values));d=math.sqrt(math.fsum(x['difference_l2_squared'] for x in values));assert math.isclose(a,metric['observed_l2'],rel_tol=1e-12) and math.isclose(b,metric['reference_l2'],rel_tol=1e-12) and math.isclose(d/b,metric['relative_l2'],rel_tol=1e-12)
assert len(full['metrics']['parameter_update']['per_tensor'])==1120;selected=full['metrics']['gradients_before_clip']['per_tensor'][MODULE+'.weight'];ratio=math.sqrt(selected['difference_l2_squared']/selected['reference_l2_squared']);assert math.isclose(ratio,diagnostic['native_vs_original_gradient']['relative_l2'],rel_tol=1e-12)
payload=W/'candidate/replay/observed.pt';assert 0<payload.stat().st_size<256*1024**2 and sha(payload)==full['observed_sha256'];summary=json.loads(read('candidate/replay/summary.json'));assert summary['observed_sha256']==full['observed_sha256'] and summary['replay_source_sha256']=='cdf2f908887db4163c29c32793491ed781fcafa2585a177e85f0ec4af75abded'
same=socket.gethostname()==OWNER['host'];alive=None
if same:
 p=Path('/proc')/str(OWNER['PID'])/'stat';alive=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==OWNER['startticks'];assert not alive
print(json.dumps(dict(passed=True,claim=CLAIM,request_id=REQUEST,result=result,accounting=rows,accounting_columns=columns,assigned=assigned,before=before,after=after,lease=lease,audit=audit,independent_operand_reference=independent,replay_summary=summary,private_evidence=evidence,observed_payload_bytes=payload.stat().st_size,observed_payload_SHA256=full['observed_sha256'],byte_identical_to_closed_default=full['observed_sha256']=='f44f0d40d198699fcd3a635bb8fc744b907dcba05d8fe2284a2967f34f0192a3',capture_file_bytes=capture.stat().st_size,capture_SHA256=sha(capture),comparison_reductions_consistent=True,original_controller_same_host_observable=same,original_controller_alive=alive,GPU_replayed=False,raw_tensor_transfer_bytes=0,Model_GO=False)))
'''
pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text();ns={'__file__':str(D/'reader_reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns);payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nOWNER='+repr(h['owned_controller'])+'\nPINS='+repr(pins)+'\nFRAMEWORK='+repr(task['framework_commit'])+'\nMODULE='+repr(task['module'])+'\n').encode()+REMOTE.encode()
with (D/'terminal-reader-intent.json').open('x') as f:json.dump(dict(original_claim=claim,original_request=request,payload_SHA256=hashlib.sha256(payload).hexdigest(),source_file_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),readonly=True,GPU_replayed=False,raw_tensor_transfer_bytes=0),f,indent=2)
try:r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=45);rc=r.returncode;stdout=r.stdout;stderr=r.stderr
except subprocess.TimeoutExpired as e:rc=None;stdout=e.stdout or b'';stderr=e.stderr or b''
for name,raw in [('terminal-reader.stdout.private.json',stdout),('terminal-reader.stderr.private.txt',stderr)]:
 with (D/name).open('xb') as f:f.write(raw)
v=json.loads(stdout) if rc==0 else None;record=dict(returncode=rc,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),value=v,stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),original_request=request,readonly=True,GPU_replayed=False)
with (D/'independent-original176-terminal.json').open('x') as f:json.dump(record,f,indent=2)
result=v.get('result',{}) if v else {};audit=v.get('audit',{}).get('diagnostic',{}) if v else {}
print(json.dumps(dict(returncode=rc,independent_passed=v.get('passed') if v else None,actual_GPU=result.get('actual_GPU_step'),accounting=v.get('accounting') if v else None,numerical_passed=result.get('numerical_passed'),native_vs_local=audit.get('native_vs_local_F64_rounded'),native_vs_original=audit.get('native_vs_original_gradient'),byte_identical_to_closed_default=v.get('byte_identical_to_closed_default') if v else None,original_controller_same_host_observable=v.get('original_controller_same_host_observable') if v else None,original_controller_alive=v.get('original_controller_alive') if v else None,Model_GO=False)))
