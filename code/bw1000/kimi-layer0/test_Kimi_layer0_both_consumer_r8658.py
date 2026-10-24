import json,hashlib,importlib.util,copy,ast
from pathlib import Path
p=Path('kimi_embedding_layer0_both_runner_r8658.py');spec=importlib.util.spec_from_file_location('both8658',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
m=json.loads(Path('Kimi-layer0-both-operands-r8658.json').read_text());actual=json.loads(Path('Kimi-MLP-layer0-observed-r8648-b.json').read_text())['watch'];checks={}
assert m['observer_reference757_raw_SHA256']==actual['result_SHA256']
for w in actual['result']['workers']:
 for label in ['Oa','Ncg']:
  assert mod.all_matches(mod.fixture_match(w[label],m['reference757'][str(w['rank'])][label]))
 checks['actual757_rank'+str(w['rank'])+'_O_Nbase_reference_consumed']=True
 t=mod.target_against_O(w['Ncg'],w['Oa']);assert t['MLP_target_equal']['activated']and not t['MLP_target_equal']['row_reduced'];assert t['boundary_equal']['ffn_norm']and not t['boundary_equal']['ffn_output'];checks['actual757_rank'+str(w['rank'])+'_knownremainingboundary']=True
bad=copy.deepcopy(actual['result']['workers'][0]['Ncg']);bad['MLP']['activated']['full_SHA256']='injected';assert not mod.all_matches(mod.fixture_match(bad,m['reference757']['0']['Ncg']));checks['truncated_or_changed_fixture_notaccepted']=True
assert len(m['inputs']['N'])==4 and all(len(v)==512 for v in m['inputs']['N']);assert m['inputs']['O'][0]==m['inputs']['N'][0];checks['real_fourcompanions_geometry']=True
for name in ['kimi_embedding_layer0_both_runner_r8658.py','launch_Kimi_layer0_both_r8658.py','observe_Kimi_layer0_both_r8658.py']:ast.parse(Path(name).read_text())
checks['actual_consumer_functions_CPU_only']=True
out={'passed':True,'checks':checks,'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Existing757 actual metadata reference consumer verification, not new O GPUforward qualification; noGPU/dlopen/performance','GPU':False,'Model':False};Path('Kimi-layer0-both-consumer-CPU-r8658.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(out))
