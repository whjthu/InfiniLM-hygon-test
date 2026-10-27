import pathlib,json,hashlib,ast,copy
p=pathlib.Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();m=json.loads((p/'GPU-manifest.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(p/n)==h,n
for f in p.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
ns={};exec(compile((p/'audit_sparse_r133.inc.py').read_text(encoding='utf8'),'audit-functions','exec'),ns)
def episode(label,rate=10.):
 return {'label':label,'serving':{'elapsed_seconds':100.,'requests':[{'stop_reason':'EOS'}for _ in range(32)],'unfinished':0},'task_scores':[{'flexible_correct':i<31,'parse_failed':False}for i in range(32)],'metrics':{'useful_visible_output_tokens':1000,'overall_output_tok_s':rate},'audit_native_events':[{'prefill':False,'native_seconds':50.}]}
es=[episode(s)for s in ['controls-before','sparse-profile','controls-after']];q=ns['analyze_envelopes'](es);assert q['perturbation_passed']and q['quality_passed'];bad=copy.deepcopy(es);bad[1]['metrics']['overall_output_tok_s']=10.4;assert not ns['analyze_envelopes'](bad)['perturbation_passed'];bad=copy.deepcopy(es);bad[1]['serving']['requests'][0]['stop_reason']='max_new_tokens';assert not ns['analyze_envelopes'](bad)['quality_passed']
pre=ast.parse((p/'preengine_r133.py').read_text(encoding='utf8'));loop=next(n for n in ast.walk(pre)if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)and [getattr(x,'id','')for x in n.target.elts]==['sym','expect']);assert all(len(t.elts)==2 for t in loop.iter.elts)
probe=(p/'glm_arrival_probe_r133.py').read_text(encoding='utf8');assert 'range(6)'in probe and "receipt['sparse_observer']=sparse_read()"in probe and "int(label=='sparse-profile')"in probe and 'time.monotonic()+900'in probe
controller=(p/'controller_r133.py').read_text(encoding='utf8');assert 'offset127'not in controller
assert 'ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'==m['Model_SHA256']
q={'source_checks_passed':True,'CPU_only':True,'GPU_or_Runtime_initialized':False,'AST_all_passed':True,'hashes_verified':len(m['source_SHA256']),'perturbation_guard_positive_and_rejection_paths_passed':True,'preengine_dladdr_pair_schema_passed':True,'native_math_or_Model_proof':False};(p/'source-checks-r133.json').write_text(json.dumps(q,indent=2),encoding='utf8');print(json.dumps(q))
