import pathlib,ast,json,tempfile,time,types,hashlib
r=pathlib.Path.cwd();p=r/'modern_Deep_owner_Model_r8925.py';s=p.read_text();tree=ast.parse(s);defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['atomic_save','checkpoint']];tail=next(n for n in reversed(tree.body)if isinstance(n,ast.Try));tail.body=[ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError',ctx=ast.Load()),args=[ast.Constant('fake_native400')],keywords=[]))];mod=ast.fix_missing_locations(ast.Module(body=defs+[tail],type_ignores=[]))
with tempfile.TemporaryDirectory()as d:
 path=pathlib.Path(d)/'report.json';report={'completed':False,'phase':'selected_CAPI_enter','phase_window':1};seen=[]
 def destroy(ctx):
  v=json.loads(path.read_text());assert v['error_phase']=='selected_CAPI_enter'and'fake_native400'in v['error']and v['shutdown_entered']and v['phase']=='shutdown_entered';seen.append('durable_error_before_destroy');raise RuntimeError('fake_destroy_failure')
 g=dict(report=report,path=path,json=json,time=time,ctx=1,check=lambda code:None,lib=types.SimpleNamespace(modern_native_engine_destroy=destroy))
 try:exec(compile(mod,str(p),'exec'),g)
 except RuntimeError as e:assert str(e)=='fake_destroy_failure'
 v=json.loads(path.read_text());assert seen==['durable_error_before_destroy']and'fake_native400'in v['error']and'fake_destroy_failure'in v['shutdown_error'];out=dict(hardware='BW1000',level='CPU_ERROR_CHECKPOINT',status='PASSED',actual_entry_AST=True,noGPU_dlopen=True,entry_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),native_error_atomic_before_destroy=True,shutdown_entered_atomic_before_destroy=True,original_error_and_phase_survive_destroy_failure=True)
q=r/'Deep-guard-Model-error-CPU-r8925.json';q.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))