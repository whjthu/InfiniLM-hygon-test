"""CPU-only model-source/interface contract; no GPU imports or controller runs."""
import copy,ctypes,hashlib,importlib.util,json,os,pathlib
D=pathlib.Path(__file__).resolve().parent;M=D/'model';BASE=D.parent/'Kimi-current-model-sparse-profile-launch-private-r136b'
spec=importlib.util.spec_from_file_location('registry_contract',M/'runtime/model_registry_contract_r162.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
os.environ['INFINILM_LOADING_BINDINGS_R9212']='1';os.environ['INFINILM_KIMI_W13_WAVEPARTIAL_R155']='0'
tests=[];q=mod.validate_loaded_registry({'loader_flushes':378,'registry_total_8x8_r159':[[2,10,378,20,1,10,378,100]for _ in range(8)]});assert q['passed'];tests.append('fixed_cache_ON_first_plus_finish_2_builds_per_owner')
for index in(0,2,4,6):
 bad=[[2,10,378,20,1,10,378,100]for _ in range(8)];bad[0][index]+=1
 try:mod.validate_loaded_registry({'loader_flushes':378,'registry_total_8x8_r159':bad})
 except AssertionError:tests.append('wrong_owner_counter_rejected_'+str(index))
 else:raise AssertionError('cache shape/count guard')
class Fn:
 def __init__(self):self.rows=[[0,0]for _ in range(8)]
 def __call__(self,d,out,n):
  for i in range(n):out[i]=self.rows[d][i]
  return 0
class Lib:pass
lib=Lib();lib.modern_kimi_W13_wavepartial_counts_r155=Fn();guard=mod.W13OffProof(lib)
for row in lib.modern_kimi_W13_wavepartial_counts_r155.rows:row[0]+=3
assert guard.finish()['passed'];tests.append('W13_baseline_only_counter_proof')
lib.modern_kimi_W13_wavepartial_counts_r155.rows[4][1]+=1
try:guard.finish()
except AssertionError:tests.append('W13_candidate_branch_rejected')
else:raise AssertionError('W13 OFF guard')
commands=[json.loads(p.read_bytes())for p in sorted((M/'queue').glob('q*.json'))];cases=[c for c in commands if c['action']=='case'];assert len(cases)==6 and [c['C']for c in cases]==[1,1,1,1,8,8]and commands[-1]['action']=='stop'
assert all(c['W']==8 and c['strategy']=='state'and c['cap']==512 and c['W2pair']and not c['model_profile']and not any('W13'in k for k in c)for c in cases)
assert len({r['id']for c in cases for r in c['requests']})==16 and sum(len(c['requests'])for c in cases)==20;tests.append('6_realEOS_pilot_cases_20_executions_16_unique')
original={q['id']:q for p in (BASE/'queue').glob('q*.json')for q in json.loads(p.read_bytes()).get('requests',[])}
assert all(r==original[r['id']]for c in cases for r in c['requests']);tests.append('original_frozen_questions_uncropped_and_answer_identity')
for rel in('runtime/batch_native_r9128.py','runtime/batch_cohort_r9128.py','runtime/target_protocol.py','runtime/task_quality_r9258.py','launcher/common.py','launcher/resources.json'):
 assert (M/rel).read_bytes()==(BASE/rel).read_bytes().replace(b'r136b',b'r162'),rel
tests.append('original_TP_PP_spec_state_quality_resource_logic_unchanged')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();manifest=json.loads((M/'runtime/model-manifest-r9183.json').read_bytes());prefix='/refactor/incoming/Kimi-sparse-profile-Model-r162/'
for path,h in manifest['source_SHA256'].items():assert sha(M/path[len(prefix):])==h
assert manifest['required_environment']['INFINILM_LOADING_BINDINGS_R9212']=='1'and manifest['required_environment']['INFINILM_KIMI_W13_WAVEPARTIAL_R155']=='0'
pins=json.loads((M/'launcher/model-source-pins.json').read_bytes())
for path,h in pins['files'].items():assert sha(M/path[len(prefix):])==h,path
assert pins['manifest_SHA256']==sha(M/'runtime/model-manifest-r9183.json')and not(M/'launcher/GPU-GO-after-source-publication.json').exists();tests.append('source_manifest_and_launcher_pins_closed_no_GPU_GO')
out={'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'dlopen':False,'controller_executed':False,'tests':tests,'model_manifest_SHA256':sha(M/'runtime/model-manifest-r9183.json'),'Native_CPU511_not_Model_qualified':True}
(D/'CPU-model-contract-observed-r162.json').write_text(json.dumps(out,indent=2));print(json.dumps({'passed':True,'checks':len(tests),'GPU':False}))
