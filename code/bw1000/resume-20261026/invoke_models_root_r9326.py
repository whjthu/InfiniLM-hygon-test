import pathlib,hashlib,sys,runpy,contextlib,io,json
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={}
for name,rel,pin,args in [('GLM','GLM-M15-Model-r129/launch_CPU_keeper_r129.py','3a6ec9ad912e3db4ffd108289179ca24502b0aa21b98719b0e546135cd733d53',['--manifest-sha','b2045541a6a75a6820860d7708f387a86cee5734e4846b18f248888bd4cfaa05','--go-file',str(r/'incoming/GLM-M15-Model-r129/root-GO-after-b588627.json')]),('Kimi','Kimi-ctx2k-Model-r127/launcher/invoke_once_r127.py','c2e328ed8ab42e6acb6496e35b7b5ad7c19ae90d2379c1ac31c00b6809c6b2b7',[])]:
 p=r/'incoming'/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==pin;sys.path.insert(0,str(p.parent));sys.argv=[str(p),*args];buf=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf):runpy.run_path(str(p),run_name='__main__')
  out[name]={'invoked':True,'result':json.loads(buf.getvalue())}
 except Exception as e:out[name]={'invoked':False,'error_type':type(e).__name__,'error':str(e)[:500]}
print(json.dumps(out))
