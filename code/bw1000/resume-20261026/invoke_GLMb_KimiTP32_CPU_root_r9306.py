import pathlib,hashlib,runpy,sys,contextlib,io,json
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={}
items=[('GLM',r/'incoming/GLM-M15-component-r121b/launch_CPU_only_r121b.py','9086d30825d3da7d55875c25c9156d55b5f490e06508f751993ed08d2793f395',['--parent','910407','--node','a14r2n20','--manifest-sha','e6238e8d119ed8ad7e9beb247b461464b719b0ea2ab0edc0d2cc1f2d6d32ae56']),('Kimi',r/'incoming/launch_CPU_r120b.py','acc8ed312ea54fd044df89160d8c67d15e7e709eb53741aa4cb888d3b7af5105',[])]
for name,p,h,args in items:
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h;sys.argv=[str(p),*args];buf=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf):runpy.run_path(str(p),run_name='__main__')
  out[name]=json.loads(buf.getvalue())
 except BaseException as e:out[name]={'launch_error':repr(e),'stdout':buf.getvalue()}
print(json.dumps(out))
