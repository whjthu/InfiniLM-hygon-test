import pathlib,hashlib,runpy,sys,json,contextlib,io
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
items=[('Kimi','incoming/Kimi-ctx2048-r123/launch_CPU_r123b.py','412860ab75272dc2f5558b1e81a007ebeb2ae5e82e066770d34b695803102946',[]),('GLM','incoming/GLM-M15-oproj-Model-r125/launch_CPU_only_r125.py','50f7997748fb4813ca943119e03901c249c9728942b0e4aee3260b97339ddefe',['--parent','910407','--node','a14r2n20','--manifest-sha','c874180c3109a37969074dfcb698d191d077849781ad5e40bf1d61a495c3b41f'])]
out={}
for name,rel,pin,args in items:
 p=r/rel
 assert hashlib.sha256(p.read_bytes()).hexdigest()==pin
 sys.argv=[str(p),*args]
 buf=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf):runpy.run_path(str(p),run_name='__main__')
  out[name]={'invoked':True,'result':json.loads(buf.getvalue())}
 except Exception as e:out[name]={'invoked':False,'error_type':type(e).__name__,'error':str(e)[:500]}
print(json.dumps(out))
