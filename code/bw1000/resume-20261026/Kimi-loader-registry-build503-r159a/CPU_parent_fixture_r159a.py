"""Local read-only fixtures using the exact retained503 receipt and actual9212d flags."""
import ast,copy,hashlib,json,pathlib
from parent_receipt_r159a import normalize,audit_symbols,PARENT_RECEIPT,PARENT_NATIVE,PRESERVED
D=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群')
wrapper=R/'resource-read-root-r9456-resume-b.json';item=json.loads(wrapper.read_bytes())['files'][2]
assert item['SHA256']==PARENT_RECEIPT
parent=item['data'];q=normalize(parent);tests=['actual503_native_exact_output_selected','33_original_objects_exactly_covered','31_unchanged_plus2_new_schema','W13_W2_math_object_SHA_preserved']
assert q['link'][q['link'].index('-o')+1]==parent['native'] and parent['native']!=parent['library']
assert parent['commands'][-1][parent['commands'][-1].index('-o')+1]==parent['library']
bad=[]
x=copy.deepcopy(parent);x['commands']=[c for c in x['commands']if not('-o'in c and c[c.index('-o')+1]==parent['native'])];bad.append(('adapter_only_link_rejected',x))
x=copy.deepcopy(parent);x['new_object_SHA256'][next(iter(PRESERVED))]='0'*64;bad.append(('changed_math_object_rejected',x))
x=copy.deepcopy(parent);x['new_object_SHA256'].pop(next(iter(PRESERVED)));bad.append(('incomplete_new_objects_rejected',x))
x=copy.deepcopy(parent);x['commands'].append(q['link']);bad.append(('ambiguous_Model_link_rejected',x))
x=copy.deepcopy(parent);x['new_MD'].pop('provider.d');bad.append(('missing_compile_MD_rejected',x))
x=copy.deepcopy(parent);x['GPU']=True;bad.append(('non_CPU_parent_rejected',x))
for name,x in bad:
 try:normalize(x)
 except (AssertionError,KeyError):tests.append(name)
 else:raise AssertionError(name)
t=json.loads((D/'compile-templates-r159a.json').read_bytes());assert t['qualified_receipt_SHA256']=='dbaedb154c3ea77e9f787b3d31bcac4fc9abb14306971315d56cbb2b24e58527'
for name,cmd in t['commands'].items():
 assert cmd[0]=='g++'and cmd[cmd.index('-c')+1].endswith('/'+name+'_r9212.cpp')and not any(x.endswith('.cu')for x in cmd)
tests.append('host_compile_commands_are_actual_qualified9212d_receipt_commands')
required=['modern_kimi_engine_load_r2475','modern_kimi_engine_finish_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_loading_registry_work_r159','modern_kimi_loading_binding_counts_r9212','modern_kimi_load_parameters_cached_r9212']
oldraw='\n'.join('000 T '+n for n in required[:3])+'\n000 W _ZN8nlohmann_vector_int_C1\n000 W _ZN8nlohmann_vector_int_C2\n'
newraw='\n'.join('000 T '+n for n in required)
audit=audit_symbols(oldraw,newraw,required[-1]);assert set(audit['removed_exports_by_type'].values())=={'W'};tests.append('JSON_vector_C1_C2_weak_removal_is_classified_not_false_gate')
for name in(required[0],required[3]):
 try:audit_symbols(oldraw,newraw.replace('000 T '+name+'\n','').replace('000 T '+name,''),required[-1])
 except AssertionError:tests.append('missing_required_symbol_rejected_'+name)
 else:raise AssertionError(name)
for f in D.glob('*.py'):ast.parse(f.read_text())
assert b'\r\n'not in(D/'CPU_inside_r159a.sh').read_bytes()
out={'passed':True,'CPU_only':True,'local_receipt_fixture_only':True,'Native_compile_link_or_dlopen':False,'tests':tests,'parent_receipt_SHA256':PARENT_RECEIPT,'parent_native_SHA256':PARENT_NATIVE,'objects':len(q['objects']),'preserved_math_objects':PRESERVED,'compile_template_source_receipt_SHA256':t['qualified_receipt_SHA256']}
(D/'CPU-parent-fixture-observed-r159a.json').write_text(json.dumps(out,indent=2));print(json.dumps({'passed':True,'checks':len(tests),'objects':len(q['objects'])}))
