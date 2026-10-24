import pathlib,json,hashlib,ast,re
r=pathlib.Path('.');d=r/'GLM-resident-exact-Model-private-r8853';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((d/'GLM-CROSSREQUEST-Model-source-manifest-r8853.json').read_text())
for n,h in m['source_SHA256'].items():
 assert sha(d/n)==h,n
 if n.endswith('.py'):ast.parse((d/n).read_text(encoding='utf-8-sig'),filename=n)
ns={};exec((d/'glm_crossrequest_contract_r8853.py').read_text(),ns);g=json.loads((d/'GLM477-frozen-gold-r8844.json').read_text(encoding='utf-8-sig'));p=g['input_ids']
for n in(16,32):
 ids=g['generated_ids'][:n];e=g['stopped_at_EOS'][:n];assert ns['validate_reference']('I512','R',p[:n],ids,e,128,{},p,{},g)[0]
 bad=[list(v)for v in ids];bad[-1][0]=(bad[-1][0]+1)%154880;assert not ns['validate_reference']('I512','R',p[:n],bad,e,128,{},p,{},g)[0]
base=(d/'glm_crossrequest_resident_base_r8853.py').read_text();pre=(d/'GLM_transient_preflight_r8853.py').read_text();audit=(d/'audit_GLM_prefill_Model_r8853.py').read_text()
assert 'BANK_BYTES=160*1024**2'in base and '(160*1024**2,6*1024**3'in pre and 'selected*160*1024**2'in audit and 'selected*256*1024**2'not in audit
assert 'RESERVE=6*GiB'in base and 'ALLOWANCE=64*1024**2'in base and 'FLOOR=512*1024**2'in base and 'self.chosen=None'in base
tree=ast.parse(audit);branch=next(x for x in ast.walk(tree)if isinstance(x,ast.If)and "after_select_QA_scope"in ast.unparse(x))
loop=ast.For(target=ast.Name(id='ep',ctx=ast.Store()),iter=ast.Constant(value=('before_select','after_select')),body=[branch,ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='visited',ctx=ast.Load()),attr='append',ctx=ast.Load()),args=[ast.Name(id='ep',ctx=ast.Load())],keywords=[]))],orelse=[]);code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),'qa-scope','exec')
for rejected in(False,True):
 q={'resident_candidate_rejected':rejected,'after_select_QA_scope':'SKIPPED_REJECTED_CANDIDATE','after_select_shared_flag':0};env={'q':q,'visited':[]};exec(code,env);assert env['visited']==(['before_select']if rejected else['before_select','after_select'])
q={'resident_candidate_rejected':True,'after_select_QA_scope':'WRONG','after_select_shared_flag':0}
try:exec(code,{'q':q,'visited':[]})
except AssertionError:pass
else:raise AssertionError('invalidQA rejection scope accepted')
worker=ast.parse((d/'modern_glm_prefill_micro_r8853.py').read_text());cleanup=next(x for x in ast.walk(worker)if isinstance(x,ast.Assert)and "report.get('cohort_phases_completed')"in ast.unparse(x));cleanupcode=compile(ast.fix_missing_locations(ast.Module(body=[cleanup],type_ignores=[])),'actual-worker-cleanup-assert','exec')
gate=next(x for x in ast.walk(tree)if isinstance(x,ast.If)and len(x.body)==1 and isinstance(x.body[0],ast.Assert)and "after_select_shared_flag"in ast.unparse(x.body[0])and "normal_numeric_rejection"in ast.unparse(x.test));gatecode=compile(ast.fix_missing_locations(ast.Module(body=[gate],type_ignores=[])),'actual-audit-afterQA-predicate','exec')
for report in[{'cohort_phases_completed':True},{'normal_numeric_rejection':True},{'resident_candidate_rejected':True,'normal_numeric_rejection':False,'cohort_phases_completed':False}]:exec(cleanupcode,{'report':report})
try:exec(cleanupcode,{'report':{}})
except AssertionError:pass
else:raise AssertionError('unclassified incomplete run passed cleanup')
for q in[{'after_select_shared_flag':1},{'resident_candidate_rejected':True,'normal_numeric_rejection':False,'after_select_shared_flag':0},{'normal_numeric_rejection':True,'after_select_shared_flag':0}]:exec(gatecode,{'q':q})
try:exec(gatecode,{'q':{'after_select_shared_flag':0}})
except AssertionError:pass
else:raise AssertionError('healthy run skipped mandatory afterQA')
names=['modern_glm_prefill_micro_r8853.py','glm_prefill_micro_probe_r8853.py','glm_crossrequest_control_r8853.py','glm_crossrequest_resident_base_r8853.py','glm_layer77_policy_r8853.py','glm_window_control_r8853.py'];symbols=set()
for n in names:symbols.update(re.findall(r'\blib\.([A-Za-z_]\w*)',(d/n).read_text()))
original=(d/'modern_glm_full_model_probe_r2.py').read_text().split('def worker(rank,payload):')[0];symbols.update(re.findall(r'\blib\.([A-Za-z_]\w*)',original))
q={'CPU_only':True,'GPU':False,'AST_source_SHA':'PASS','geometry_gold_B16_B32_positive_negative':'PASS','exact160MiB_allcontrols':'PASS','QA_rejection_skip_and_bad_scope_negative':'PASS','actual_worker_cleanup_predicate_positive_negative':'PASS','actual_audit_afterQA_predicate_positive_negative':'PASS','same_single_fixedmask_and_originalreserve':'PASS','actual_consumer_CAPI_names':sorted(symbols),'manifest_SHA256':sha(d/'GLM-CROSSREQUEST-Model-source-manifest-r8853.json')};(r/'GLM-resident-exact-CPUchecks-r8853.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
