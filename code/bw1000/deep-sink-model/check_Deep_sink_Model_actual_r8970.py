"""Read-only host CPU closure; never dlopen/GPU/Model."""
import pathlib,json,hashlib,subprocess,ast
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=r/'incoming/Deep-sink-Model-manifest-r8970.json';m=json.loads(mp.read_text())
def actual(p):
 p=pathlib.Path(p)
 return r/p.relative_to('/refactor')if str(p).startswith('/refactor/')else p
for group,base in [('incoming_source_SHA256',r/'incoming'),('existing_incoming_dependency_SHA256',r/'incoming'),('existing_remote_SHA256',r)]:
 for n,h in m[group].items():assert sha(base/n)==h,(group,n)
bp=r/m['cpu_receipt_path'];q=json.loads(bp.read_text());assert q['completed']and q['CPU_only']and q['producer']=='910407.220';lib=actual(q['libraries']['full']['path']);assert sha(lib)==m['DSO_SHA256'];objects={p:sha(actual(p))for p in q['link']['full']['objects']};assert objects==q['link']['full']['objects']
nm=subprocess.check_output(['nm','-D','--defined-only','-C',str(lib)],text=True);required=q['actual_Model_consumer_required_symbols']['required'];missing=[n for n in required if not any(v.split()[-1]==n for v in nm.splitlines())];assert not missing;shared={k:[s for s in nm.splitlines()if s.endswith('deep_owner_r8900::'+k)]for k in ['pool','active','window']};assert all(len(v)==1 for v in shared.values())
u=q['new_compiled_units'];assert len(u)==1 and u[0]['source_SHA256']=='54cf4654cd56e55421cd4b0ec0779d8e0f74e4ce310ae7ec71ea6005590ea166';assert sha(actual(u[0]['source']))==u[0]['source_SHA256']and sha(actual(u[0]['object']))==u[0]['object_SHA256'];deps={p:sha(actual(p))for p in u[0]['deps']if str(p).startswith('/refactor/')};assert all(h==u[0]['deps'][p]for p,h in deps.items())
inside=r/'incoming/Deep_sink_Model_r8970_inside.sh';subprocess.run(['bash','-n',str(inside)],check=True,timeout=15);entry=r/'incoming/modern_Deep_sink_Model_r8970.py';es=entry.read_text();protocol=r/'incoming/Deep_sink_Model_protocol_r8970.py';ps=protocol.read_text();launch=(r/'incoming/launch_Deep_sink_Model_r8970.py').read_text();[ast.parse(v)for v in [es,ps,launch]]
assert 'modern_Deep_sink_Model_r8970.py'in inside.read_text()and 'Deep_sink_Model_r8970_inside.sh'in launch and 'Deep-sink-Model-manifest-r8970.json'in launch and 'from Deep_sink_Model_protocol_r8970 import run'in es
assert es.index("cpu_receipt_path=r/manifest['cpu_receipt_path']")<es.index("prebuild['completed']")<es.index('torch.cuda.device_count')<es.index('lib=ctypes.CDLL')
assert "[(0,'A0'),(1,'C1'),(1,'C2')]"in ps and 'deep_owner_arm_r8900'not in ps and 'deep_owner_read_r8900'not in ps and 'deep_model_quant1_mode_r8886(0)'in ps and 'deep_reference_scratch_mode_r8953(1)'in ps and 'deep_reference_sink_mode_r8967(arm)'in ps
assert 'remaining>=400'in ps and 'SKIPPED_BUDGET'in ps and 'packet_started=time.perf_counter()'in es
assert 'deep_latent_prefill_mode_r8933(0)'in ps and 'accepted_decode_fused_counter_delta'in ps and '5461'in ps
assert 'shutdown_entered'in es and 'error_phase'in es and 'prefill_CAPI_enter_'in ps and 'scratch_actual_Tblock_path'in ps and 'prefill_only_no_decode_hits'in ps
out=dict(hardware='BW1000',level='CPU_ACTUAL_CONSUMER',status='PASSED',CPU_only=True,no_GPU_dlopen=True,manifest_SHA256=sha(mp),entry_SHA256=sha(entry),protocol_SHA256=sha(protocol),CPU_receipt_path=m['cpu_receipt_path'],CPU_receipt_SHA256=sha(bp),DSO_SHA256=sha(lib),actual_required_symbols_count=len(required),missing_symbols=missing,shared_guarded_collector_actual_nm=shared,actual_link_object_count=len(objects),all_actual_link_objects_read_SHA256=objects,candidate_unit_actualsource_SHA256=u[0]['source_SHA256'],candidate_unit_actualobject_SHA256=u[0]['object_SHA256'],refactor_dependency_actualread_count=len(deps),before_CUDA_metadata=True,actual_launch_inside_entry_protocol_chain=True,new_native_compile=False,owner_capture=False)
print(json.dumps(out))
