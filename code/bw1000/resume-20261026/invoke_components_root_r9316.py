import pathlib,json,hashlib,sys,runpy,contextlib,io
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={}
items=[('GLM','GLM-M15-offset8-r127','launch_CPU_keeper_r127.py','ed1197503e3f3bb38953d57ce1275b9fca2f8352f81796af2fd65507cdee1c73','67a303e9102a725c0ceb6be6bb58fbac8918b8103884647eb8e4dd2b9fa3595b'),('TP32','Kimi-TP32-I96-GPU-controller-r120e','launch_CPU_keeper_r120e.py','32039dd7cad5869bcb307dccb3dc8c814e0fe09f923b6a99cd80a52369e90219','80a98cba1b6d54c094a6f16c2962908ceeeafc2a8f206c0b054a37d8a0937748')]
for name,folder,filename,pin,manifest in items:
 p=r/'incoming'/folder/filename;assert hashlib.sha256(p.read_bytes()).hexdigest()==pin
 sys.path.insert(0,str(p.parent));sys.argv=[str(p),'--manifest-sha',manifest,'--go-file',str(p.parent/'root-GO-after-e196867.json')];buf=io.StringIO()
 try:
  with contextlib.redirect_stdout(buf):runpy.run_path(str(p),run_name='__main__')
  out[name]={'invoked':True,'result':json.loads(buf.getvalue())}
 except Exception as e:out[name]={'invoked':False,'error_type':type(e).__name__,'error':str(e)[:500]}
print(json.dumps(out))
