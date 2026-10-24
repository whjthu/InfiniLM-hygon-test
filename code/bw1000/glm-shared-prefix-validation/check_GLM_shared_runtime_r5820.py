"""CPU-only check actual generated source/plan/counter expressions; no Torch."""
import pathlib,json,hashlib,ast,types
r=pathlib.Path(__file__).resolve().parent;checks=0
def ok(v):
 global checks
 assert v;checks+=1
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((r/'GLM-CROSSREQUEST-Model-source-manifest-r5819.json').read_text())
for n,h in m['source_SHA256'].items():
 ok(sha(r/n)==h)
 if n.endswith('.py'):ast.parse((r/n).read_text())
for n in ['launch_GLM_crossrequest_Model_r5819.py','watch_GLM_transient_Model_terminal_r5819.py','audit_GLM_crossrequest_Model_r5819.py']:ast.parse((r/n).read_text())
p=(r/'glm_layer77_probe_r5819.py').read_text();tree=ast.parse(p)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
nodes=[n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in ['perf_labels','labels']for t in n.targets)]
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual phase plan>','exec'),ns)
ok(len(ns['perf_labels'])==8 and len(ns['labels'])==9)
for repeat in (1,2):ok([s.split('-')[2]for s in ns['perf_labels']if f'-r{repeat}-'in s]==['O','N','N','O'])
ok("generate(label,i512_inputs,8,True,128"in p and 'model_window'not in p and 'window.before'not in p)
ok("policy.switch(label,int(prefill_on),native_diag,shadow)"in p and all(not s.startswith('validate-')for s in ns['labels']))
ok("'1'if shared_on else'0'"in p and p.index('shared_before=shared_stats()')<p.index('start = time.perf_counter()'))
gen=next(n for n in ast.walk(fn)if isinstance(n,ast.FunctionDef)and n.name=='generate')
start=next(i for i,n in enumerate(gen.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='shared_delta'for t in n.targets))
end=next(i for i,n in enumerate(gen.body[start:],start)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and isinstance(n.value.func.value,ast.Name)and n.value.func.value.id=='row'and n.value.func.attr=='update')
block=compile(ast.Module(body=gen.body[start:end+1],type_ignores=[]),'<actual counter audit>','exec')
results=[]
for L,F in [(38,10),(40,10)]:
 for on in [False,True]:
  events=[{'round':step,'rows':list(range(group*8,group*8+8))}for step in range(128)for group in (0,1)]
  skip=(L-F)*16*127 if on else 0;fall=(L-F)*16*128-skip
  env=dict(shared_stats=lambda:[skip,fall],shared_before=[0,0],events=events,prompts=[[1]*512 for _ in range(16)],stage_layers=L,full_layers=F,shared_on=on,cache_policy=1,row={},label='CPU')
  exec(block,env);ok(env['row']['shared_validation_counter_delta']==[skip,fall])
  ok(env['row']['shared_validation_source_exact_avoided_D2H_bytes']==((L-F)*16*sum(range(513,640))*8 if on else 0))
  results.append({'L':L,'F_example_only':F,'ON':on,**env['row']})
  env['shared_stats']=lambda:[skip+1,fall]
  try:exec(block,env)
  except AssertionError:checks+=1
  else:raise AssertionError('incorrect native counter must reject')
pre=(r/'GLM_transient_preflight_r5819.py').read_text();pe=(r/'GLM_transient_preengine_r5819.py').read_text();entry=(r/'launch_GLM_crossrequest_Model_r5819.py').read_text()
ok('candidates/glm-shared-validation-model-r5817/libglm_shared_validation_model_r5817.so'in pe)
ok("('modern_glm_shared_validation_counts_r5814',private)"in pe)
ok('4df185ee83bc7706e524d99ffc24ac151142f003334a700d6db1197b82b013da'in pre)
ok("'00:45:00'"in entry and 'watch.poll()is not None'in entry and entry.index("watch=subprocess.Popen")<entry.index("controller=subprocess.Popen"))
cpu=json.loads((r/'r5818-GLM-shared-CPU-a.json').read_text())['CPU-build.json'];ok(cpu['completed']and cpu['fresh_M_equal_MD']and cpu['ABI']==1)
out={'passed':True,'checks':checks,'scope':'Actual AST phase plan and actual counter block CPU execution/source-SHA/ABI path closure; no nativeGPU/quality/performance claim. NativeCPU184 passed separately. F=10 table entries are fixtures, actual runtime F comes from exact model config.','counter_fixtures':results,'native_CPU':{'producer':cpu['producer'],'library_SHA256':cpu['library_SHA256'],'receipt_SHA256':cpu['SHA256']}}
dest=r/'GLM-shared-runtime-CPU-r5820.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2),encoding='utf8');print('PASSED',checks,sha(dest))
