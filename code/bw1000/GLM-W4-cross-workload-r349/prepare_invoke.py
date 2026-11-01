import pathlib,json
D=pathlib.Path(__file__).parent;go=json.loads((D/'private-execution-GO.prepublish.json').read_bytes());go['publication_commit']='40830260b66c639d853665d2611f9c4d2438b117'
source="""import pathlib,json,subprocess,sys,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/GLM-W4-cross-workload-r349/controller_v3';go=GO
p=r/'results/GLM349-execution-GO-40830260.json'
assert not p.exists()
with p.open('x')as f:json.dump(go,f,indent=2);f.flush();os.fsync(f.fileno())
z=subprocess.run([sys.executable,'-B',str(d/'invoke_once_r349.py'),'--execution-go',str(p)],capture_output=True,text=True,timeout=90)
if z.returncode:print(json.dumps({'launcher_failed':True,'returncode':z.returncode,'stdout':z.stdout[-5000:],'stderr':z.stderr[-5000:],'no_replay':True}))
else:
 q=json.loads(z.stdout.strip().splitlines()[-1]);q['execution_GO_path']=str(p);print(json.dumps(q))
""".replace('go=GO','go='+repr(go));compile(source,'remote_invoke_once.py','exec');(D/'remote_invoke_once.py').write_bytes(source.encode())
