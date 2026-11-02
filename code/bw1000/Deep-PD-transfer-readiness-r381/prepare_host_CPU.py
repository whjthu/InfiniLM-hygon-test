from pathlib import Path
import ast,json,hashlib,base64
G=Path(__file__).resolve().parents[3];L=G.parent;D=Path(__file__).with_name('host-v1');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'state_wire.py')=='85d8dea5f3cbfc2c5240163ade7edb2ce9f2e73abdadc83b36d8ae6a491afdc8'
plan=dict(schema=1,GPU_GO=False,Model_GO=False,Native_dlopen=False,actual_KV_state=False,scope='One publishedCPU2node run with one TCPconnection, one ticket containing8syntheticU8 hostblocks; allshards checksum+hostpatternverify before one publish/ACK. Not actualKVlayout or Nativeimport qualification; no bandwidth/Modelgain claim.',
 wire=dict(origin='code/bw1000/Deep-PD-disaggregation-r382/state_wire.py',origin_commit='380a34bf4ff5c04be7073e200d5ef97cbd8f787e',SHA256=sha(D/'state_wire.py'),unchanged_copy=True),
 CPU_claim=dict(parent=910376,nodes=['a02r4n12','a02r4n13'],tasks=2,CPUs_per_node=1,host_mem_MiB_per_node=4096,Slurm_hard_s=120,original_worker_deadline_s=90,srun_wait_s=110,report_visibility_bound_s=30,entry_max_s=150,GPU_cards=0,one_attempt=True,no_model_weights=True),
 route=dict(P='12.1.2.72',D='12.1.2.73',interface='ib0 IPoIB',transport='TCP hostmemory, no deviceRDMA',port=49181,single_connection_no_retry=True),
 payload=dict(dtype='uint8',blocks=[dict(shard=i,bytes=(1<<20)+17*i)for i in range(8)],total_bytes=sum((1<<20)+17*i for i in range(8)),shape='Syntheticprotocolfixture only, not DeepKVgeometry'),
 timings='Own-node monotonic source export-through-whole-ticketACK and destination receive-verify-publish only. No crossnode subtraction and no measuredKV/Modelthroughput inference.',
 lifecycle='CPUreadonlyKFDprecheck, oneTCPaccept/connect, all8block stage/checksum+hostpatternverify/publish+ACK, close createdsockets and releasehoststaging; no actualTP8worker orCCL teardown qualification',
 local_CPU_qualification='Actualsameworker execute + actualCPUentry sourceverify/recordingSlurmargv/env + realTCPloopback; success, corruptpattern allshardsnonpublish, originaldeadline, duplicateintent. Not actualSlurm or Native.')
(D/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
files=['worker.py','CPU_main.py','check_CPU.py','state_wire.py','PLAN.json']
for n in files:
 if n.endswith('.py'):ast.parse((D/n).read_bytes())
mf=dict(schema=1,GPU_GO=False,files_SHA256={n:sha(D/n)for n in files},wire_origin_commit=plan['wire']['origin_commit'],wire_origin_SHA256=plan['wire']['SHA256'],CPU_claim=plan['CPU_claim'])
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(mf,indent=2)+'\n',encoding='utf8');h=sha(D/'SOURCE_MANIFEST.json')
task_path=G/'coordination/tasks/Deep-PD-transfer-readiness-r381.json';t=json.loads(task_path.read_bytes());t.update(state='HOST_V1_CPU_SOURCE_READY_NOT_SUBMITTED',host_CPU_source='code/bw1000/Deep-PD-transfer-readiness-r381/host-v1',host_CPU_manifest_SHA256=h,host_CPU_claim=plan['CPU_claim'],host_CPU_submitted=False);task_path.write_text(json.dumps(t,indent=2)+'\n',encoding='utf8')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in files+['SOURCE_MANIFEST.json']}
remote="from pathlib import Path\nimport json,base64,hashlib\nD=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Deep-PD-transfer-readiness-r381-host-v1');D.mkdir(exist_ok=False)\npayload="+repr(payload)+"\nfor n,b in payload.items():(D/n).write_bytes(base64.b64decode(b))\nsha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()\nassert sha(D/'SOURCE_MANIFEST.json')=="+repr(h)+"\nm=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==v for n,v in m['files_SHA256'].items())\nprint(json.dumps(dict(uploaded=True,GPU=0,CPU_submitted=False,source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'))))\n"
ast.parse(remote);(L/'remote381-host-v1-upload.py').write_text(remote,encoding='utf8');print(json.dumps(dict(source_manifest_SHA256=h,files=files,GPU=0,CPU_submitted=False)))
