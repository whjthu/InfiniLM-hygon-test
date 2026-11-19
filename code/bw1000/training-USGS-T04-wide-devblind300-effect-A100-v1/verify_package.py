"""Stdlib byte/AST verification only; no original data, fixtures, imports or devices."""
import ast
import hashlib
import json
from pathlib import Path
D=Path(__file__).resolve().parent

def need(ok,why):
    if not ok:raise ValueError(why)

def verify(budget=lambda:None):
    mf=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    actual={p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    need(actual==set(mf)|{'SOURCE_FILES.json'},'EXACT_OUTER_INVENTORY')
    for name,pin in mf.items():
        budget();p=D/name;raw=p.read_bytes()
        need(not p.is_symlink() and len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'EXACT_SOURCE_PIN')
        if p.suffix=='.py':ast.parse(raw)
    plan=json.loads((D/'INSTALL_PLAN.json').read_bytes());claim=json.loads((D/'CLAIM.json').read_bytes())
    need(plan['label']==claim['label']==D.name=='training-USGS-T04-wide-devblind300-effect-A100-v1'
         and plan['namespace']==claim['namespace']=='/home/whj/nvidia-baseline/alignment-assets/'+D.name,'EXACT_NEW_V2_SCOPE')
    need(plan['expected_GPU']==dict(index=5,UUID='GPU-19c8d012-5264-d041-a10c-48aa94938099',PCI='00000000:A7:00.0',total_MiB=40960,minimum_free_MiB=38000)
         and plan['master_port']==29935 and claim['runtime_device_fallback'] is False,'FIXED_GPU5_NO_FALLBACK')
    need(plan['cooperative_lock']==claim['resources']['cooperative_lock']=='/home/whj/nvidia-baseline/alignment-assets/gpu-cooperative-locks/'+plan['expected_GPU']['UUID']+'.lock'
         and claim['resources']['fixed_physical_index']==5 and claim['resources']['fixed_UUID']==plan['expected_GPU']['UUID'],'GPU5_CLAIM_LOCK')
    need(plan['CPU_affinity']==claim['resources']['fixed_CPU_affinity']==[6,7,8,9],'FIXED_CPU6_9_NO_FALLBACK')
    need(hashlib.sha256((D/'core/SOURCE_FILES.json').read_bytes()).hexdigest()==plan['core_manifest_SHA256'],'CORE_MF')
    flags=[]
    for name in ['CLAIM.json','INSTALL_PLAN.json','core/PLAN.json','core/CLAIM.json']:
        x=json.loads((D/name).read_bytes());flags.extend([x['root_GO'],x['GPU_GO']])
    need(all(type(x)is bool and x==flags[0] for x in flags),'EIGHT_MATCHING_GO_FLAGS')
    need(claim['max_invocations']==1 and not claim['Model_GO'] and not claim['training_GO'] and not claim['formal17'],'DEV_DIAGNOSTIC_ONLY')
    need(hashlib.sha256((D/'qualified/bounded_GPU_supervisor.py').read_bytes()).hexdigest()==plan['qualified_driver_SHA256']=='2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f','QUALIFIED_RSS_HELPER')
    c=json.loads((D/'core/PLAN.json').read_bytes());f=json.loads((D/'core/FRAMEWORK_BINDING.json').read_bytes())
    need(len(f['files'])==404 and c['initial_named_SHA256']=='086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137','SAME_INITIAL_ORIGINAL404')
    need(c['native_identities']['bytes']==256132 and c['native_identities']['SHA256']=='dc259a06262ab55fa3aa72d854d97a40367a70d8c0af14ba7e20f15ef9dbf679','ACTUAL_NATIVE150_REFERENCE')
    need(c['data_binding']['bytes']==98073 and c['data_binding']['SHA256']=='9eafc89f673d5c7b468ac30594fbe71a535b9137e6ccc5f738755e14f56b3540','ACTUAL_NEW_DATA_BINDING')
    need(c['native150_public']['bytes']==11107 and c['native150_public']['SHA256']=='1b10a52aad5f73722dbec24f9580344f33228a498a3fe49c09f4e055f8ff798e','ACTUAL_PUBLISHED_NATIVE150')
    need(c['planned_generations']==claim['scope']['real_generations']==300 and c['per_arm']==100
         and c['split_denominators']=={'dev':50,'blind':50} and len(c['media'])==20,'FIXED300_SCOPE')
    need(c['inputs']['dev']=={'bytes':32020,'SHA256':'f3019f9845f5379a2afc6239607ecb162a8a01ff012c6420e9c19859f361c601','rows':50}
         and c['inputs']['blind']=={'bytes':32120,'SHA256':'29cf1894e1c01964dbb8e301e99e8008bd1115049455e955f4d61008d55cbc3c','rows':50},'EXACT_NEW_TWO_INPUT_FILES')
    for name,pin in c['public_prerequisites'].items():
        raw=(D/'core/prerequisites'/name).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'PUBLISHED_PREREQUISITE')
    for name,pin in [('response_protocol.py',c['content_parser']),('EVALUATION_PROTOCOL.json',c['protocol_pin'])]:
        raw=(D/'core'/name).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'FROZEN_NEW_PROTOCOL')
    need(c['content_parser']['SHA256']=='5dce77a93bc557bb3c6aee479ca4775d5862190896ecfca5c68f171cb65f98c6','PREREGISTERED_PARSER')
    need(claim['clock']['whole_seconds']==plan['whole_seconds']==c['whole_epoch_cap_seconds']==5400
         and claim['clock']['native_deadline_offset']==5250 and claim['clock']['supervisor_stop_offset']==5300
         and claim['clock']['held_and_registry_cleanup_offset']==5360 and claim['clock']['ordinary_SSH_client_offset']==5395,'ONE_CLOCK_OFFSETS')
    need(plan['CPU_count']==4 and plan['RSS_soft_bytes']==64*1024**3
         and plan['host_MemAvailable_admission_bytes']==32*1024**3 and plan['host_MemAvailable_runtime_stop_bytes']==8*1024**3,'RESOURCE_SCOPE')
    origin=json.loads((D/'ORIGIN.json').read_bytes())
    for pin in origin['unchanged_members']:
        raw=(D/pin['path']).read_bytes();need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'],'UNCHANGED_QUALIFIED_HELPER_OR_MATH')
    reader=ast.parse((D/'read_original.py').read_bytes())
    cp=[n.value.value for n in reader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REVIEWED_CALLER_SHA256' for t in n.targets)]
    need(cp==[hashlib.sha256((D/'core/infer_wide100.py').read_bytes()).hexdigest()],'PRE_REGISTERED_CALLER_PIN')
    need(c['paired25_training_metadata_qualified'] is False and c['domain_original_training_accepted'] is False and c['domain25']['original_training_accepted'] is False and c['original_T04_domain_training_failure_retained'] is True,'NO_ORIGINAL_T04_FAILURE_RECLASSIFICATION')
    need(c['initial']['bytes']==174808783 and c['initial']['SHA256']=='b574c04a3bdaaff7f061d7285dc0fc73f2cbadce4801fca97dbd00ad8772fa0d','ACTUAL_T04_INITIAL_NOT_T03_CAPTURE')
    need(c['domain25']['files']['adapter_model.safetensors']==dict(bytes=174663096,SHA256='6391597e38c5cbf568c65cba51066e798006c806f5bba7606ac022143c9f8510') and c['generic25']['files']['adapter_model.safetensors']==dict(bytes=174663096,SHA256='8f3ca18172f7a40d03ec4f45759b8bff0f3848e254361fc48117eadf1bd55f15'),'ACTUAL_T04_CHECKPOINTS')
    b=json.loads((D/'core/T04-native-binding.json').read_bytes())
    need(c['trainable_scope']['tensors']==len(b['trainable_manifest'])==504 and c['trainable_scope']['parameters']==sum(r['numel'] for r in b['trainable_manifest'])==43646976,'T04_TYPED_PARAMETER_SCOPE')
    worker=ast.parse((D/'core/infer_wide100.py').read_bytes())
    calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_model_processor'];need(len(calls)==1,'ONE_LOADER_CALLSITE')
    kws={x.arg:x.value.value for x in calls[0].keywords if isinstance(x.value,ast.Constant)}
    need(kws['model_type']=='llava_onevision1_5' and kws['attn_impl']=='eager' and kws['load_model'] is True and kws['download_model'] is False,'ACTUAL_T04_LOADER_ROUTE')
    need(c['input_pin_state']=='ACTUAL_T04_NATIVE150_DB_FILES_PINNED' and c['saved25_checkpoint_values_qualified'] is True,'T04_NATIVE_AND_STATE_BOUND')
    return dict(passed=True,files=len(mf),root_GO=flags[0],GPU_GO=flags[0],
        source_manifest_SHA256=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest(),
        actual_model=0,actual_remote=0,actual_GPU=0,fixture_executions=0)

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
