from pathlib import Path
import hashlib,json,subprocess
B=Path(r'C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7');D=B/'training-sdxl-bf16-reduction-v3';h=json.loads((D/'transport-outcome.json').read_bytes())['value'];pre=(B/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8');ns={'__file__':str(D/'reuse.py')};exec(compile(pre.split('REMOTE=',1)[0],'ordinary_cluster_ssh','exec'),ns)
code='''from pathlib import Path
import hashlib,json,math,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-sdxl-bf16-reduction-v3')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
intent=json.loads((W/'intent.json').read_bytes());assert intent['request_id']==REQUEST and intent['claim_commit']==CLAIM
result=json.loads((W/'controller-result.json').read_bytes());assert result['completed'] and result['request_id']==REQUEST and result['actual_GPU_step']=='1087656.169'
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=12).splitlines() if '|training-sdxl-bf16-reduction-v3' in x]
assert [(x[0],x[2],x[3]) for x in rows]==[(x['terminal']['row'][0],x['terminal']['row'][2],x['terminal']['row'][3]) for x in result['runs']]
logs={}
for receipt in result['runs']:
 p=W/(receipt['name']+'.log');assert sha(p)==receipt['log_SHA256'];logs[p.name]=dict(bytes=p.stat().st_size,SHA256=receipt['log_SHA256'])
gpu=result['GPU_receipt'];assert gpu['returncode']==0 and not gpu['forced_cleanup'] and gpu['terminal']['row'][2:4]==['COMPLETED','0:0']
before=json.loads((W/'before.json').read_bytes());after=json.loads((W/'after.json').read_bytes());lease=json.loads((W/'lease-before.json').read_bytes());assigned=json.loads((W/'assigned.json').read_bytes());assert before['node']==after['node']==lease['node']==assigned['node']=='f11r4n11';assert before['pci']==after['pci']==lease['pci'] and len(before['pci'])==8 and before['boot_id']==after['boot_id']==lease['boot_id']==assigned['boot_id'];assert before['kfd_holder_count']==after['kfd_holder_count']==lease['kfd_holder_count']==0;assert assigned['step']==lease['step']=='1087656.169' and assigned['visible_cards']==1 and assigned['pci']=='0000:09:00.0' and assigned['KFD']['gpu_id']==22078 and not assigned['assigned_PCI_pending'] and not assigned['KFD']['other_devices_gpu_id_read']
policy=json.loads((W/'candidate/policy.json').read_bytes());assert policy['executed_update'] and policy['original_policy'] is True and policy['observed_policy'] is False and policy['restored_policy'] is True and policy['replay_source_SHA256']=='cdf2f908887db4163c29c32793491ed781fcafa2585a177e85f0ec4af75abded'
raw=(W/'comparison.json').read_bytes();full=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==result['comparison']['comparison_SHA256'] and full['checks']==result['comparison']['checks'] and full['passed']==result['numerical_passed'];assert full['reference_sha256']=='c59a71095924edfdfb9ad7041b8ef32f747e7825557a313f72ad09a3b0b53036' and full['policy_sha256']=='075e86e54a3eeefb8d5aad219a23f16959d14bd42b22103cd5fe5302c823f6b5'
p=W/'candidate/replay/observed.pt';assert 0<p.stat().st_size<=256*1024**2 and sha(p)==full['observed_sha256']
summary=json.loads((W/'candidate/replay/summary.json').read_bytes());assert summary['observed_sha256']==full['observed_sha256']
for key,metric in full['metrics'].items():
 if 'per_tensor' not in metric:continue
 values=list(metric['per_tensor'].values());a=math.sqrt(math.fsum(x['observed_l2_squared'] for x in values));b=math.sqrt(math.fsum(x['reference_l2_squared'] for x in values));d=math.sqrt(math.fsum(x['difference_l2_squared'] for x in values));assert math.isclose(a,metric['observed_l2'],rel_tol=1e-12) and math.isclose(b,metric['reference_l2'],rel_tol=1e-12) and math.isclose(d/b,metric['relative_l2'],rel_tol=1e-12)
owner=json.loads((W/'keeper-owner.json').read_bytes());alive=None
if socket.gethostname()==owner['host']:
 p=Path('/proc')/str(owner['PID'])/'stat';alive=p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==owner['startticks'];assert not alive
print(json.dumps(dict(passed=True,result=result,accounting=rows,accounting_columns=result['accounting_columns'],assigned=assigned,before=before,after=after,lease=lease,policy=policy,replay_summary=summary,logs=logs,original_controller_alive=alive,comparison_reductions_consistent=True,observed_payload_bytes=(W/'candidate/replay/observed.pt').stat().st_size,private_evidence={n:dict(bytes=(W/n).stat().st_size,SHA256=sha(W/n)) for n in ('controller-result.json','assigned.json','lease-before.json','before.json','after.json','candidate/policy.json','candidate/replay/summary.json','comparison.json','reused-original-policy.json','reused-host-guard-tests.json')},GPU_replayed=False,raw_tensor_transfer_bytes=0)))
'''
payload=('CLAIM='+repr(h['claim_commit'])+'\nREQUEST='+repr(h['request_id'])+'\n'+code).encode();(D/'terminal-read-intent.json').write_bytes((json.dumps(dict(original_claim=h['claim_commit'],original_request=h['request_id'],payload_SHA256=hashlib.sha256(payload).hexdigest(),read_only=True,GPU_replayed=False),indent=2)+'\n').encode())
r=subprocess.run(ns['ssh'],input=payload,capture_output=True,timeout=45);(D/'terminal-read-stdout.private.json').write_bytes(r.stdout);(D/'terminal-read-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0,r.stderr[-300:];q=json.loads(r.stdout);(D/'independent-original169-terminal.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());s=q['result'];print(json.dumps(dict(passed=q['passed'],actual_GPU=s['actual_GPU_step'],numerical_passed=s['numerical_passed'],comparisons=s['comparison'],policy=q['policy'],accounting=q['accounting'],peak_reserved_bytes=q['replay_summary']['peak_reserved_bytes'],original_controller_alive=q['original_controller_alive'])))
