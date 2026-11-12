"""Consume saved native evidence while preserving the original overall FAIL."""
from pathlib import Path
import ast,hashlib,json,subprocess,time
D=Path(__file__).resolve().parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
i=json.loads((D/'launch-intent.json').read_bytes());claim=i['claim'];request=i['request_id']
t=json.loads(subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':coordination/tasks/'+D.name+'.json']))
sources={}
for n,pin in t['source_SHA256'].items():
 raw=subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+t['source']+'/'+n]);assert hashlib.sha256(raw).hexdigest()==pin;sources[n]=raw
files={target:dict(bytes=len(sources[n]),SHA256=hashlib.sha256(sources[n]).hexdigest()) for target,n in t['remote_files'].items()}
s=sources['read_original_terminal.py'].decode()
changes={
 "assert o['returncode']==0 and o['passed'] and o['owned_group_reaped'] and o['stop_reason'] is None":"assert o['returncode']==1 and o['passed'] is False and o['owned_group_reaped'] and o['stop_reason'] is None",
 "assert r['passed'] and r['GPU_cards']==1 and r['full_native_pipeline_loaded'] and r['actual_preset_fusion_executed']":"assert r['passed'] is False and r['GPU_cards']==1 and r['phase']=='private-initial-state-and-provider-identities' and r['error_type']=='AssertionError'",
 "assert r['base_tensors']==521 and r['preset_targets']==240 and r['actual_trainable_tensors']==476 and r['actual_trainable_elements']==9922560":"assert artifacts['failure-private.json']['value']['traceback'].endswith('AssertionError\\n') and 'assert allocated_after==0 and reserved_after==0' in artifacts['failure-private.json']['value']['traceback']",
 "assert r['frozen_DiT_tensors']==521 and r['frozen_DiT_elements']==6154908736 and r['allocated_after']==r['reserved_after']==0":"# Pre-exit allocated/reserved values were not persisted;remain UNKNOWN.",
 "assert r['actual_forward_calls']==r['actual_backward_calls']==r['actual_optimizer_updates']==0":"# Frozen executed worker source reaches no forward/backward/optimizer calls.",
 "assert artifacts['FUSION_IDENTITY.json']['SHA256']==r['FUSION_IDENTITY_SHA256']":"assert artifacts['FUSION_IDENTITY.json']['SHA256']==o['FUSION_IDENTITY.json']['SHA256'] and fusion==o['FUSION_IDENTITY.json']['value']",
 "assert target_count==240 and 0<changed<=240 and changed==fusion['changed_target_tensors']==r['changed_target_tensors']":"assert target_count==240 and changed==240 and changed==fusion['changed_target_tensors']",
 "assert artifacts['IMPORTED_SOURCE_FILES.json']['SHA256']==r['source_provider_manifest_SHA256'] and len(providers)==r['source_provider_files']":"assert 40<=len(providers)<=200",
 "assert artifacts['trainable-initial.private.safetensors']['SHA256']==r['initial_trainable_SHA256']":"assert artifacts['trainable-initial.private.safetensors']['bytes']>19845120",
 "assert artifacts['trainable-initial.private.safetensors']['bytes']==r['initial_trainable_bytes']":"""p=W/'trainable-initial.private.safetensors'
 with p.open('rb') as stream:
  length=int.from_bytes(stream.read(8),'little');assert 0<length<1024**2
  header=json.loads(stream.read(length));header.pop('__metadata__',None)
 assert set(header)==set(expected) and len(header)==476
 ranges=[]
 for name,pin in header.items():
  assert pin['dtype']=='BF16' and pin['shape']==expected[name]['shape']
  ranges.append(pin['data_offsets'])
 ranges.sort();cursor=0
 for begin,end in ranges:assert begin==cursor and end>begin;cursor=end
 assert cursor==19845120 and cursor+8+length==p.stat().st_size""",
 "assert artifacts['post-constructor-RNG.private.pt']['SHA256']==r['RNG_SHA256']":"assert artifacts['post-constructor-RNG.private.pt']['bytes']<1024**2",
 "assert r['independent_numerical_backend'] is False and not r['numerical_training_qualified'] and not r['formal_performance'] and not r['Model_GO']":"assert not r['Model_GO']",
 "artifacts=artifacts,post=post,source_files=len(EXPECTED_FILES),reader_original_weight_bytes_read=0,reader_private_initial_values_exported=False,":"artifacts=artifacts,post=post,source_files=len(EXPECTED_FILES),reader_original_weight_bytes_read=0,reader_private_initial_values_exported=False,original_run_passed=False,original_returncode=1,original_failure_preserved=True,partial_native_fusion_qualified=failure is None,preexit_allocator_values=None,preexit_allocator_assertion_failed=True,"
}
for old,new in changes.items():assert s.count(old)==1,old;s=s.replace(old,new)
ast.parse(s);(D/'read_partial_original_terminal.py').write_bytes(s.encode())
payload=('CLAIM='+repr(claim)+'\nREQUEST='+repr(request)+'\nEXPECTED_FILES='+repr(files)+'\nGPU_UUID='+repr(t['GPU_UUID'])+'\nGPU_PCI='+repr(t['GPU_PCI'])+'\n').encode()+s.encode()
start=time.monotonic();r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-27','python3 -B -u -'],input=payload,capture_output=True,timeout=110)
(D/'partial-reader.stdout.private.json').write_bytes(r.stdout);(D/'partial-reader.stderr.private.txt').write_bytes(r.stderr)
v=json.loads(r.stdout) if r.stdout else None
result=dict(transport_rc=r.returncode,RTT_seconds=time.monotonic()-start,claim=claim,request_id=request,
 reader_source_SHA256=hashlib.sha256(s.encode()).hexdigest(),stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),value=v,new_GPU_invocations=0)
with (D/'partial-reader-result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(transport_rc=r.returncode,reader_passed=v.get('passed') if v else None,failure=v.get('failure') if v else None,
 original_run_passed=False,original_returncode=1,partial_native_fusion_qualified=v.get('partial_native_fusion_qualified') if v else None,
 post=v.get('post') if v else None,new_GPU_invocations=0)))
