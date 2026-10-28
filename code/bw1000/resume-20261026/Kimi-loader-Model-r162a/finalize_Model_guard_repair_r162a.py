from pathlib import Path
import ast,hashlib,json,sys,tempfile,shutil,datetime,subprocess
V=Path(__file__).resolve().parent;D=V/'Kimi-loader-Model-private-r162a';M=D/'model';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mp=M/'runtime/model-manifest-r9183.json';m=json.loads(mp.read_bytes())
m['matrix']={'unique_question_ids':16,'request_executions':20,'case_count':6,'C1_cases':4,'C8_cases':2,'C1_is_subset_of_C8':True,'not_new32_heldout':True,'nativeC':8,'width':8,'strategy':'state','profile':'OFF; loading CPU/fault/IO/registry attribution only','allcards':33,'C16':'deferred capacity8 retained','epsilon':0,'projectedcache':False,'loading_registry_cache':'fixedON','W13':'fixedOFF'}
mp.write_bytes((json.dumps(m,indent=2)+'\n').encode())
gp=M/'launcher/GPU-GO-schema.json';g=json.loads(gp.read_bytes());g['manifest_SHA256']=sha(mp);gp.write_bytes((json.dumps(g,indent=2)+'\n').encode())
prefix='/refactor/incoming/Kimi-sparse-profile-Model-r162a/'
pin=M/'launcher/model-source-pins.json';producer=M/'producer-source-r162a.json'
pins=json.loads(pin.read_bytes());pins['manifest_SHA256']=sha(mp);pins['files']={prefix+f.relative_to(M).as_posix():sha(f)for f in M.rglob('*')if f.is_file()and f not in (pin,producer)};pin.write_bytes((json.dumps(pins,indent=2)+'\n').encode())
prod=json.loads(producer.read_bytes());prod['manifest_SHA256']=sha(mp);prod['matrix']=json.loads((M/'fixed-data-r162a.json').read_bytes());prod['files_SHA256']={f.relative_to(M).as_posix():sha(f)for f in M.rglob('*')if f.is_file()and f!=producer};producer.write_bytes((json.dumps(prod,indent=2)+'\n').encode())
subprocess.run([sys.executable,'-B','-X','utf8',str(D/'CPU_model_contract_r162a.py')],check=True)
sys.path.insert(0,str(M/'launcher'));from pilot_queue_contract import validate_frozen_pilot
tests=[]
with tempfile.TemporaryDirectory()as temp:
    T=Path(temp);shutil.copytree(M/'queue',T/'queue');proof=json.loads((M/'fixed-data-r162a.json').read_bytes());(T/'fixed-data-r162a.json').write_text(json.dumps(proof))
    assert validate_frozen_pilot(T)['passed'];tests.append('actual_keeper_contract_good_queue')
    for key,value in [('case_count',12),('unique_question_ids',32),('registry_cache_fixed_ON',False)]:
        bad=dict(proof);bad[key]=value;(T/'fixed-data-r162a.json').write_text(json.dumps(bad))
        try:validate_frozen_pilot(T)
        except AssertionError:tests.append('reject_'+key)
        else:raise AssertionError('failed mutation check '+key)
    (T/'fixed-data-r162a.json').write_text(json.dumps(proof));q=T/'queue/q0000.json';orig=q.read_bytes();bad=json.loads(orig);bad['requests'][0]['id']='outside-c8-set';q.write_text(json.dumps(bad))
    try:validate_frozen_pilot(T)
    except AssertionError:tests.append('reject_C1_not_subset')
    else:raise AssertionError('failed C1subset mutation')
    q.write_bytes(orig)
tree=ast.parse((M/'launcher/model_keeper.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate')
assert isinstance(fn.body[0],ast.ImportFrom)and fn.body[0].module=='pilot_queue_contract'
assert isinstance(fn.body[1],ast.Assign)and fn.body[1].value.func.id=='validate_frozen_pilot';tests.append('keeper_uses_exact_tested_contract_before_other_preflights')
for p in M.rglob('*.py'):ast.parse(p.read_text())
assert all(b'\r\n'not in p.read_bytes()for p in M.rglob('*.sh'))
out={'passed':True,'CPU_only':True,'GPU':False,'Runtime':False,'controller_executed':False,'tests':tests,'manifest_matrix_matches_actual_pilot':True,'old_failed_handle':'910407.515','old_GPU_submitted':False,'Native_recompiled':False}
(D/'CPU-guard-regression-r162a.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
shutil.copyfile(V/'prepare_Model_guard_repair_r162a.py',D/'prepare_Model_guard_repair_r162a.py');shutil.copyfile(Path(__file__),D/'finalize_Model_guard_repair_r162a.py')
outer={'sealed_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_only':True,'GPU_submitted':False,'launcher_executed':False,'Model_qualified':False,'actual_four_node_CPU_preflight':'PENDING_ROOT_EXECUTION','CPU511_native_SHA256':m['native_DSO_SHA256'],'model_manifest_SHA256':sha(mp),'source_pins_SHA256':sha(pin),'producer_SHA256':sha(producer),'CPU_interface_contract_SHA256':sha(D/'CPU-model-contract-observed-r162a.json'),'guard_regression_SHA256':sha(D/'CPU-guard-regression-r162a.json'),'review_SHA256':sha(D/'MODEL-REVIEW.zh-CN.md'),'export_model_files_SHA256':{f.relative_to(M).as_posix():sha(f)for f in M.rglob('*')if f.is_file()},'files_SHA256':{f.relative_to(D).as_posix():sha(f)for f in D.rglob('*')if f.is_file()and f.name!='source-manifest-r162a.json'},'actual_GO_present':False,'entry':'model/launcher/preflight_then_run.py'}
outerpath=D/'source-manifest-r162a.json';outerpath.write_bytes((json.dumps(outer,indent=2)+'\n').encode())
print(json.dumps({'passed':True,'regression_checks':len(tests),'outer_SHA256':sha(outerpath),'Model_manifest_SHA256':sha(mp),'producer_SHA256':sha(producer),'source_pins_SHA256':sha(pin),'export_model_files':len(outer['export_model_files_SHA256']),'GPU':False}))
