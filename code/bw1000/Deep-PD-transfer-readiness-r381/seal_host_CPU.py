"""Audit one actual host-memory transaction; no Native/KV/device performance claim."""
from pathlib import Path
import json,hashlib,gzip,base64
G=Path(__file__).resolve().parents[3];L=G.parent;sha=lambda b:hashlib.sha256(b).hexdigest()
raw_path=L/'Deep381-host-v1-terminal-original.json';post_path=L/'Deep381-host-v1-post-original.json'
raw=json.loads(raw_path.read_bytes());post=json.loads(post_path.read_bytes());P,D,C=[raw['reports'][r]['data']for r in ['P','D','CPU']]
assert raw['source_manifest_SHA256']=='199d68e2d4fded0333a0efdc12820405585d8ccd842ec171cdcda038848eeb0b'
for v in raw['reports'].values():assert sha((json.dumps(v['data'],indent=2)+'\n').encode())==v['SHA256']
assert P['passed']and D['passed']and C['passed']and C['srun_returncode']==0
assert P['actual_step']==D['actual_step']==C['actual_CPU']=='910376.833'
assert P['all8shards_publish_ACK_received']and D['staged_shards']==8 and D['whole_ticket_host_publish_count']==1
assert D['all_host_staging_released']and P['normal_socket_close']and D['normal_socket_close']and D['host_pattern_verify_passed']
assert P['GPU']==D['GPU']==C['GPU']==0 and not P['Native_verify_called']and not D['Native_verify_called']
# Independent scalar digest recomputation from the specified byte formula, not imported producer.
hashes=[sha(bytes((j+i*37)%256 for j in range((1<<20)+17*i)))for i in range(8)]
assert P['transfer']['block_SHA256']==D['transfer']['block_SHA256']==hashes
assert P['payload_bytes']==D['payload_bytes']==8389084
for step in ['910376.833','910376.834']:
 found=[s.split('|')for s in raw['accounting']if s.startswith(step+'|')];assert len(found)==1 and found[0][2:4]==['COMPLETED','0:0']
assert post['returncode']==0 and len(post['records'])==2
assert all(not r['kfd_pids']and not r['matching_worker_PIDs']and not r['TCP_entries']and r['port49181_bindable']for r in post['records'])
out=G/'results/bw1000/2026-11-02';snap=dict(terminal=raw,post=post,transport_first_observed=json.loads((L/'Deep381-host-v1-actual-CPU-first.json').read_bytes()))
b=(json.dumps(snap,indent=2)+'\n').encode();sealed=out/'Deep381-host-v1-CPU833-original-scalar-sealed.json'
sealed.write_text(json.dumps(dict(schema=1,content='Original CPU-only hostprotocol P/D/controller reports, sourcehash/accounting and CPU cleanup. No KV/model/key/activation.',uncompressed_SHA256=sha(b),uncompressed_bytes=len(b),encoding='base64(gzip)',payload=base64.b64encode(gzip.compress(b,mtime=0)).decode()),indent=2)+'\n',encoding='utf8')
result=dict(schema=1,completed=True,host_TCP_protocol_PASS=True,GPU=0,Native_dlopen=False,actual_KV_state=False,Model_gain='UNKNOWN',device_RDMA_qualified=False,
 actual_CPU='910376.833',actual_post='910376.834',CPU_srun_returncode=0,source_commit='4b73e1504f925d0319f1a2532413f0efdadba770',source_manifest_SHA256=raw['source_manifest_SHA256'],source_files_SHA256=raw['source_files_SHA256'],
 original_reports=[dict(role=k,path=v['path'],SHA256=v['SHA256'])for k,v in raw['reports'].items()],
 route=dict(P_node=P['node'],D_node=D['node'],P_IP=P['local_IP'],D_IP=P['peer_IP'],transport='TCP over explicitly bound IPoIB ib0 hostmemory, no GPU buffers',connections=1),
 transaction=dict(shards=8,payload_bytes=8389084,bytes_per_shard=D['host_written_bytes_per_shard'],all8_digests_independently_recomputed=True,block_SHA256=hashes,all8_staged_before_host_pattern_verify=True,publish_count=1,source_ACK_for_whole_ticket=True,Native_verify=False,host_staging_released=True),
 timing=dict(source_transport_through_whole_ticket_ACK_wall_s=P['transfer']['source_transport_and_ACK_wall_s'],destination_receive_verify_publish_wall_s=D['transfer']['destination_receive_verify_publish_wall_s'],P_complete_entry_wall_s=P['local_complete_entry_wall_s'],D_complete_entry_wall_s=D['local_complete_entry_wall_s'],meaning='Own-node enclosures only; includes hashing/patternverify and sourcegeneration. Fullentry includes setup/marker/accept/connect; do not subtract clocks acrossnodes or callthis KVtransfer/devicebandwidth/PDModel performance.',startup_unisolated='About30seconds occur outside wiretransaction. Marker/storage/listen/connect/other setup not separately timestamped; exactcause UNKNOWN.'),
 terminal_accounting=raw['accounting'],post=post,
 pre_post_SSH_failure=dict(phase='Before remoteexec, SSH kex_exchange_identification connectionreset',CPU_post_submitted=False,repair='One subsequent readonly post completed; original transfer not replayed'),
 sealed=dict(path=str(sealed.relative_to(G)),SHA256=sha(sealed.read_bytes())),
 next='Transportconsumer ready to accept producer read_block hostbytes and transactional sink; actual Native export/import/complete state layout and resume must be separately proven. No GPU submitted or Modelreload authorized bythis receipt.')
p=out/'Deep381-host-v1-CPU833-transport-and-cleanup-PASS.json';p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
task_path=G/'coordination/tasks/Deep-PD-transfer-readiness-r381.json';t=json.loads(task_path.read_bytes());t.update(state='HOST_CPU833_TRANSPORT_AND_CLEANUP_PASS_GPU_NATIVE_UNKNOWN',host_CPU_submitted=True,actual_host_CPU='910376.833',actual_host_post='910376.834',host_CPU_result=str(p.relative_to(G)),host_CPU_result_SHA256=sha(p.read_bytes()),host_CPU_resources_released=True,GPU_GO=False,GPU_submitted=False,timer_or_network_transfer_started=True,host_network_transfer_started=True,GPU_timing_started=False);task_path.write_text(json.dumps(t,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(completed=True,GPU=0,actual_CPU='910376.833',result=str(p),SHA256=sha(p.read_bytes()),sealed_bytes=sealed.stat().st_size)))
