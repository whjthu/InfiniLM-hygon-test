"""Source/AST/MF/frozen arithmetic only; never calls zlib codec, fixture or private asset."""
import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent

def verify(root=D):
    mf=(root/'SOURCE_FILES.json').read_bytes();pins=json.loads(mf)['files'];actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==set(pins)|{'SOURCE_FILES.json'}
    for name,pin in pins.items():
        p=root/name;b=p.read_bytes();assert not p.is_symlink() and len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['SHA256']
        if p.suffix=='.py':ast.parse(b,filename=name)
    p=json.loads((root/'PLAN.json').read_bytes());c=json.loads((root/'CLAIM.json').read_bytes())
    assert p['schema']==c['schema']=='INTERNVL_THREE_SHARD_ZLIB_24H_V2' and p['case']==c['case']=='T03.sft'
    for k in ['root_GO','provision_GO','READ_GO','assembly_GO','GPU_GO','Model_GO','training_GO']:assert type(c[k])is bool and p[k] is c[k]
    assert not any(c[k] for k in ['assembly_GO','GPU_GO','Model_GO','training_GO']) and c['max_invocations']==1
    assert p['whole_seconds']==c['whole_seconds']==p['budget']['whole_end']==86400
    assert p['block_bytes']==1048576 and p['frame_header_bytes']==128 and p['compressed_slack_bytes']==65536 and p['chunk_bytes']==65536
    assert [r['file_id'] for r in p['files']]==[2,3,4]
    assert [(r['parent'],r['node']) for r in p['files']]==[('1087656','a14r3n04'),('1087656','a14r3n12'),('1103606','f13r1n14')]
    assert [r['source_CPU_affinity'] for r in p['files']]==[[10,11],[12,13],[14,15]] and p['source_masks_exclude']==list(range(10))
    expected=[(4915962480,'1dd4d7629deff34807654ca198f45166a532fdbc1bbda3fd80e5e69eebdcdcd4'),(4915962496,'595f3b9043d046ae612f2dbf7cbfcd2465a1c001fcb58e93f78dcaf7cc5976e1'),(2318463688,'1935bfbc7774be5e64adf7ecec5cbefc7c18dae24d498ba4609301c3423b21fd')]
    assert [(r['bytes'],r['SHA256']) for r in p['files']]==expected
    assert sum(r['bytes'] for r in p['files'])==p['raw_bytes']==c['raw_bytes']==12150388664
    assert sum(r['blocks'] for r in p['files'])==p['total_blocks']==c['block_denominator']==11590
    assert sum(r['max_wire_bytes'] for r in p['files'])==p['wire_cap_bytes']==c['wire_cap_bytes']==12911434424
    assert all(r['blocks']==(r['bytes']+1048575)//1048576 and r['max_wire_bytes']==r['bytes']+r['blocks']*(65536+128) and r['target_payload'].startswith(p['modelroot']+'/') for r in p['files'])
    assert p['budget']==dict(prepare_end=60,all_install_admission_end=180,payload_stop=84600,step_end_max=85020,host_stop=85200,held_cleanup=85500,producer_accounting_stop=85800,client_SSH_end=85920,fresh_read_stop=86280,fresh_held_cleanup=86310,consumer_closure=86340,PC_cleanup=86370,whole_end=86400)
    assert p['source_CPU']==p['target_worker_CPU']==2 and p['target_alloc_CPU']==4 and p['target_memory_MiB']==4096 and p['target_GRES']==0 and p['progress_seconds']==60 and p['progress_bytes']==67108864
    assert p['raw_fallback'] is p['replay_allowed'] is False and p['old14_accesses']==p['initial362_364_accesses']==p['assembly_actions']==0
    assert hashlib.sha256((root/'held_child.py').read_bytes()).hexdigest()=='06ef9978e7c0957abb2fd0fc6204e16e88a433a966475d53a727bf797ee85351'
    old=json.loads((root/'V1_SOURCE_PINS.json').read_bytes())['members']
    for n in ['common.py','endpoint.py','protocol.py','held_child.py','install_endpoint.py','node_entry.py']:
        b=(root/n).read_bytes();assert len(b)==old[n]['bytes'] and hashlib.sha256(b).hexdigest()==old[n]['SHA256']
    prior=json.loads((root/'PRIOR_SCOPE.json').read_bytes())
    assert prior['original_request_id']=='328b0f23c38949f1a9149f0006e48b9b' and prior['original_outcome']=='TERMINAL_FAILED'
    assert prior['original_receivers_terminal']==prior['original_sources_PID_absent']==prior['original_target_hosts_PID_absent']==3
    assert [v['SHA256'] for v in prior['public_records']]==['70b426798a4c3a78ee70adffe56b93999c9aede7624627fbaff3dd0b74718fa5','174cee0cbe6759d357edeac4c7d5435d390f39a94b2a0d180c1da0512e3c5021']
    assert p['label']=='training-internvl-three-shard-zlib-24h-v2' and p['common_root'].endswith('/'+p['label'])
    assert p['control_evidence']['actual_step_queries_max']==c['step_visibility_queries_per_receiver_max']==3
    assert p['control_evidence']['actual_step_query_timeout_seconds_max']==5 and p['control_evidence']['actual_step_query_group_seconds_max']==c['step_visibility_seconds_per_receiver_max']==15
    assert p['failed_consumer_host_metadata']['files']==['outcome.json','control-state.json','compute-ready-observed.json']
    assert p['failed_consumer_host_metadata']['aggregate_per_host_max']==c['new_host_metadata_per_arm_bytes_max']==196608
    assert c['independent_fresh_SSH_max']==c['independent_sacct_max']==3
    # Parse the embedded stdlib reader without running it or any query/fixture.
    fa=ast.parse((root/'fresh_accounting.py').read_bytes())
    remote=next(n.value.value for n in fa.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets))
    ast.parse(remote,filename='fresh_accounting.REMOTE');assert len(remote.encode())+2048<=16384
    assert sum((root/n).stat().st_size for n in [*pins,'SOURCE_FILES.json'])<=196608
    return dict(passed=True,files=len(pins),AST_files=len(list(root.glob('*.py'))),root_GO=c['root_GO'],provision_GO=c['provision_GO'],READ_GO=c['READ_GO'],MF_SHA256=hashlib.sha256(mf).hexdigest(),actual_body_reads=0,actual_remote=0,fixture_executions=0)

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
