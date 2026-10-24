import pathlib
r=pathlib.Path(__file__).resolve().parent
s=(r/'check_GLM_exact_budget_Model_r8855.py').read_text().replace('GLM-exact-budget-Model-private-r8855','GLM-M16-Model-private-r8862').replace('r8855','r8862')
s=s.replace("receipt=json.loads((r/'GLM-exact-budget-CPU-observed-r8854-a.json').read_text(encoding='utf-8-sig'))['receipt']", "receipt=json.loads((r/'GLM-M16-CPU-observed-r8861-b.json').read_bytes())['receipt'];budget=json.loads((r/'GLM-exact-budget-CPU-observed-r8854-a.json').read_bytes())['receipt']['resident_exact_budget']")
s=s.replace("receipt['resident_exact_budget']['resident_bank_bytes']","budget['resident_bank_bytes']")
s=s.replace("'GLM-exact-budget-Model-CPUchecks-r8862.json'","'GLM-M16-Model-CPUchecks-r8862.json'")
s=s.replace("'budget':'45minhard2600inner new8855;1549sactual558firstpolicy+B16~146+B32~300+QA/reset/shutdownmargin;noextraarm/retry/extension'", "'budget':'45minhard2600inner proposed B32M8/M16 one load;561historical33:02 mixedB16/B32 plus~150s equivalent additionalB32;M16time/peakUNKNOWN;noextraarm/retry/extension'")
start=s.index('linked=json.loads(');end=s.index('symbols=set()',start)
s=s[:start]+"linked=json.loads((r/'GLM-M16-actual-metadata-r8862.json').read_bytes());assert linked['passed']and linked['actual_current_DSO_setter_160MiB_per_layer']and linked['actual_metadata']['library_SHA256']==receipt['library_SHA256'];chain={'SHA256':linked['actual_current_setter_disassembly_SHA256']}\n"+s[end:]
needle="(r/'GLM-M16-Model-CPUchecks-r8862.json').write_bytes"
insert='''
# Run actual current protocol expressions with synthetic event metadata only.
probe_ast=ast.parse((d/'glm_prefill_micro_probe_r8862.py').read_bytes())
micro_guard=next(x for x in ast.walk(probe_ast)if isinstance(x,ast.Assert)and ast.unparse(x.test)=='prefill_micro == 8 and micro in (8, 16)')
guard_code=compile(ast.fix_missing_locations(ast.Module(body=[micro_guard],type_ignores=[])),'actual-M16-micro-guard','exec')
for micro in(8,16):exec(guard_code,{'prefill_micro':8,'micro':micro})
for pm,micro in[(4,8),(8,32)]:
 try:exec(guard_code,{'prefill_micro':pm,'micro':micro})
 except AssertionError:pass
 else:raise AssertionError('unfrozen micro accepted')
for n in(1,4,7,8,9,15,16):
 events=[{'round':1,'rows':list(range(n)),'flat_tokens':n}]
 got=base.expected_dispatch(events,40)
 assert got=={'append_D2D_bytes':8192*n,'exact_shared_calls':int(n in(8,16)),'resident_original601_row_calls':0 if n in(8,16)else n}
audit_ast=ast.parse((d/'audit_GLM_prefill_Model_r8862.py').read_bytes())
native_expected=next(x for x in ast.walk(audit_ast)if isinstance(x,ast.Assign)and any(isinstance(v,ast.Name)and v.id=='expected'for v in x.targets)and 'flat in (4, 8, 16)'in ast.unparse(x))
native_code=compile(ast.fix_missing_locations(ast.Module(body=[native_expected],type_ignores=[])),'actual-M16-native-count-expected','exec')
for flat in(4,8,16,7,9,15,4096):
 env={'flat':flat,'S':40};exec(native_code,env);assert env['expected']==([0]*5 if flat in(4,8,16)else[80,40 if flat>=512 else 0,40 if flat>=512 else 0,0 if flat>=512 else 80,0])
out['actual_M8_M16_micro_guard_positive_negative']='PASS'
out['actual_resident_dispatch_M8_M16_and_EOS1_4_7_9_15']='PASS'
out['actual_auditor_native_fallback_vs_vector4_8_16']='PASS'
out['native_CPU_contract']=receipt['CPU_contract']
out['B32_both_arms_same_original477_gold']=True
'''
assert needle in s;s=s.replace(needle,insert+'\n'+needle)
(r/'check_GLM_M16_Model_r8862.py').write_bytes(s.encode())
print('prepared actual-source CPU checks')
