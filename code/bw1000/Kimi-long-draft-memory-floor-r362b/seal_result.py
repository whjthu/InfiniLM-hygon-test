"""Seal actual one-draft reports and scope the pressure-recovery conclusion."""
from pathlib import Path
import hashlib,json

D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent;O=G/'results/bw1000/2026-11-02'
q=json.loads((L/'Kimi362b-terminal-independent-post.json').read_bytes())
watch,pressure,worker=[x['report']for x in q['files']]
assert watch['completed']and watch['passed']and pressure['passed']and worker['passed']
assert q['independent_post']['passed']and worker['normal_protocol_close']
assert worker['phase_counts']==dict(begin=1,append_prompt=64,seal=1,query=1,retire=1,close=1)
assert worker['feature_bytes']==32760*35840*2
assert len(worker['memory_fences'])==66 and all(x['floor_passed']for x in worker['memory_fences'])
trims=[x for x in worker['memory_fences']if x['trimmed']];assert len(trims)==1
trim=trims[0];assert trim['index']==53 and trim['free_before_bytes']<4<<30<=trim['free_after_bytes']
assert trim['allocated_before_bytes']==trim['allocated_after_bytes']
assert trim['free_after_bytes']-trim['free_before_bytes']==trim['reserved_before_bytes']-trim['reserved_after_bytes']
manifest=D/'SOURCE_MANIFEST.json';m=json.loads(manifest.read_bytes())
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
sealed=[]
for f,name in zip(q['files'],['watch','pressure','worker']):
    raw=(json.dumps(f['report'],indent=2)+'\n').encode();assert hashlib.sha256(raw).hexdigest()==f['SHA256']
    p=O/('Kimi362b-GPU662-'+name+'-sealed.json');p.write_bytes(raw)
    sealed.append(dict(path=str(p.relative_to(G)).replace('\\','/'),SHA256=f['SHA256'],remote_path=f['path']))
audit=O/'Kimi362b-GPU662-terminal-independent-post.json';audit.write_bytes((json.dumps(q,indent=2)+'\n').encode())
summary=dict(schema=1,state='DRAFT_ONLY_GEOMETRY_MEMORY_PROTOCOL_PASS_NORMAL_CLEAN',run=watch['run'],
    source_manifest_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(),runtime_members_unchanged=True,
    source_claim_commit=watch['publication_commit'],CPU705_receipt_SHA256='4aa65ebb87bf055dbd8d372c8751731254025be166ea3bd8e6aa747679009ef9',
    actual_GPU='910371.662',actual_CPU_keeper='910407.708',GPU_cards=1,target_cards=0,
    GPU_accounting_elapsed_s=60,CPU_keeper_elapsed_s=73,pressure_driver_wall_s=pressure['wall_s'],accounting=q['accounting'],
    actual_device=worker['actual_device'],strict_loader_receipt=worker['loader_receipt'],load_wall_s=worker['load_wall_s'],
    fixture=dict(synthetic=True,seed=324,base_rows=512,periodic_rows=32760,raw_dim=35840,dtype='BF16',
                 full512_SHA256=pressure['fixture_full512_SHA256'],tail504_SHA256=pressure['fixture_tail504_SHA256'],
                 feature_bytes=worker['feature_bytes'],target_activation_replay=False),
    phases=worker['phase_counts'],commands=worker['commands'],query_positions=8,query_proposals=7,
    finite_input_projection_hidden_logits_and_valid_integer_proposals=True,all66_publication_or_query_fences_passed_original4GiB_floor=True,
    pressure_recovery=dict(trim_count=1,append_ordinal=54,snapshot=trim,
        returned_inactive_cache_bytes=trim['reserved_before_bytes']-trim['reserved_after_bytes'],
        allocated_bytes_unchanged=True,guard_reduced=False,append_or_request_retried=False,
        context_projection_concat_query_math_unchanged=True,
        causal_scope='In this new synthetic geometry run at the same54thappend, releasing allocator cache restored driverfree while activeallocation stayed identical; this is not a replay of original653 targetfeatures'),
    memory=dict(minimum_observed_free_before_pressure_bytes=worker['minimum_observed_free_draft_bytes'],
                minimum_successful_posttrim_fence_free_bytes=worker['minimum_free_draft_bytes'],
                peak_allocated_bytes=worker['peak_allocated_draft_bytes'],peak_reserved_bytes=worker['peak_reserved_draft_bytes'],
                peak_scope='Torchallocator lifetime stats; fence driverfree samples are not continuous transient peaks'),
    cost_only=dict(complete_actual8position_query_RPC_wall_s=next(x['wall_s']for x in pressure['commands']if x['cmd']=='query'),
                   all64_prompt_append_RPC_wall_s=sum(x['wall_s']for x in pressure['commands']if x['cmd']=='append_prompt'),
                   performance_gain_qualified=False),
    normal_protocol_close=worker['normal_protocol_close'],worker_exit_code=pressure['worker_exit_code'],
    controller_post=watch['post'],independent_post=q['independent_post'],all_resources_released=True,
    old_failures_preserved=['run-c653memoryfloor/544cancel/694fail','preCPUuploadparse0CPU0GPU','CPU704recordingcontextfailure'],
    acceptance_qualified=False,Model_gain_qualified=False,new_target_reload_authorized=False,
    decision='Retain the minimal private one-trim fence implementation. It passed actual strictloader,32760geometry,8positionquery and normalclose. A later target4request decision requires a separate claim; none started.',
    sealed_reports=sealed,terminal_audit=dict(path=str(audit.relative_to(G)).replace('\\','/'),SHA256=hashlib.sha256(audit.read_bytes()).hexdigest()))
p=O/'Kimi362b-GPU662-memory-pressure-PASS-clean.json';p.write_bytes((json.dumps(summary,indent=2)+'\n').encode())
task=G/'coordination/tasks/Kimi-long-draft-memory-floor-r362b.json';t=json.loads(task.read_bytes())
t.update(state=summary['state'],GPU=0,GPU_GO=False,actual_GPU='910371.662',actual_CPU_keeper='910407.708',all_resources_released=True,
         terminal_result=str(p.relative_to(G)).replace('\\','/'),single_trim_recovery_bytes=summary['pressure_recovery']['returned_inactive_cache_bytes'],
         new_target_reload_authorized=False,next='Stop after this once-only draft card; root owns any subsequent independent targetclaim.')
task.write_bytes((json.dumps(t,indent=2)+'\n').encode())
print(json.dumps(dict(result=str(p),trim=trim,post=q['independent_post'])))
