"""Explicit247-to327 receipt schema comparison; never invent a linked-object key."""
from pathlib import Path
import hashlib,importlib,json,sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_bundle(directory,digest):
 directory=Path(directory);assert sha(directory/'SOURCE_MANIFEST.json')==digest
 m=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():
  assert not Path(name).is_absolute()and'..'not in Path(name).parts
  assert sha(directory/name)==h,name
 return m

def import_exact(name,path):
 value=importlib.import_module(name)
 if Path(value.__file__).resolve()!=Path(path).resolve():raise ValueError('Unexpected import provider: '+name)
 return value

def verify_artifacts(directory,gate):
 D=Path(directory);gate.check('source-and-receipt-verification');pins=json.loads((D/'bindings.json').read_bytes())
 basepath=D/'baseline-manifest-r252.json';assert sha(basepath)==pins['baseline_manifest_SHA256'];m=json.loads(basepath.read_bytes())
 # Validate source bytes before importing even the immutable contract helper.
 for path,h in m['source_SHA256'].items():assert sha(path)==h,path
 frontend=Path(m['frontend_directory']);sys.path.insert(0,str(frontend))
 contract=import_exact('entry_contract_r9143',frontend/'entry_contract_r9143.py');assert contract.validate(m)=={'tp_nodes':2,'pp':2,'logical_capacity':1}
 model,base=contract.verify_inputs(m)
 cp=Path(pins['candidate_receipt']);assert sha(cp)==pins['candidate_receipt_SHA256'];candidate=json.loads(cp.read_bytes())
 assert candidate['actual_CPU']=='910407.639'and candidate['completed']and candidate['passed']and candidate['CPU_only']and not candidate['GPU']and not candidate['weights']and not candidate['Native_dlopen']
 assert candidate['native_SHA256']==pins['candidate_native_SHA256']and sha(candidate['native'])==pins['candidate_native_SHA256']
 assert candidate['baseline_native_SHA256']==base['native_SHA256']==m['native_DSO_SHA256']==pins['baseline_native_SHA256']
 assert candidate['baseline_linked_objects']==base['all_linked_object_SHA256']
 assert candidate['replaced_HOST_objects']==pins['replacements']and len(candidate['replaced_HOST_objects'])==2
 for old,v in candidate['replaced_HOST_objects'].items():assert old in base['all_linked_object_SHA256']and sha(v['path'])==v['SHA256']
 assert candidate['actual_dependency_library_SHA256']==base['actual_dependency_library_SHA256']
 for value in candidate['actual_dependency_library_SHA256'].values():assert sha(value['path'])==value['SHA256']
 assert candidate['retained_public_CAPI_count']==130 and set(candidate['new_public_CAPI'])=={'modern_kimi_long_features_begin_r305','modern_kimi_long_features_capabilities_r305'}and candidate['dynamic_dependency_names_unchanged']
 for name,v in pins['packages'].items():verify_bundle(v['path'],v['manifest_SHA256'])
 evidence=pins['worker_qualification'];assert sha(evidence['CPU_receipt'])==evidence['CPU_receipt_SHA256']and sha(evidence['worker_report'])==evidence['worker_report_SHA256']
 cq=json.loads(Path(evidence['CPU_receipt']).read_bytes());wq=json.loads(Path(evidence['worker_report']).read_bytes())
 assert cq['completed']and cq['passed']and cq['GPU']==0 and cq['Native_dlopen']is False
 assert cq['source_manifest_SHA256']==wq['source_manifest_SHA256']==pins['packages']['Kimi-long-draft-memory-floor-r362b']['manifest_SHA256']
 assert wq['actual_step']==evidence['actual_GPU']=='910371.662'and wq['completed']and wq['passed']and wq['normal_protocol_close']
 assert wq['phase_counts']==dict(begin=1,append_prompt=64,seal=1,query=1,retire=1,close=1)and len(wq['memory_fences'])==66 and all(x['floor_passed']and x['floor_bytes']==4<<30 for x in wq['memory_fences'])
 publisher=Path(pins['packages']['Kimi-long-target-publisher-r335']['path']);driver=Path(pins['packages']['Kimi-long-pair-driver-r337']['path'])
 sys.path[:0]=[str(driver),str(publisher)]
 modules={'entry_contract_r9143':contract}
 for name in ['stage_control_r2386','parameter_loader_r9143','weight_key_guard_r9156','shard_head_contract_r9183','memory_guard_r9144','model_affinity_check_r2628','long_codec_r179','model_event_observer_r135']:
  modules[name]=import_exact(name,frontend/(name+'.py'))
 for name in ['long_capture_native','target_native_base','publisher','batch_wire_r9128','batch_policy_r9128']:
  modules[name]=import_exact(name,publisher/(name+'.py'))
 for name in ['pair_driver','capture_transport','grade_output']:
  modules[name]=import_exact(name,driver/(name+'.py'))
 return dict(manifest=m,baseline=base,candidate=candidate,model=model,modules=modules,frontend=frontend,driver=driver,
  binding_input={'native':candidate['native']}, # not a fabricated CPUreceipt
  audit=dict(baseline_receipt_SHA256=m['native_receipt_SHA256'],candidate_receipt_SHA256=pins['candidate_receipt_SHA256'],baseline_native_SHA256=base['native_SHA256'],candidate_native_SHA256=candidate['native_SHA256'],baseline_object_count=len(base['all_linked_object_SHA256']),replaced_HOST_objects=candidate['replaced_HOST_objects'],retained_public_CAPI=130,SDK_dependencies_unchanged=True,package_manifests=pins['packages']))
