import pathlib,sys,json,base64,subprocess
D=pathlib.Path(__file__).resolve().parent;L=D.parents[3];R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def remote(code):
 args=['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile='+str(L/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -']
 p=subprocess.run(args,input=code,capture_output=True,text=True,timeout=190)
 assert p.returncode==0,(p.stdout,p.stderr)
 return json.loads(p.stdout)
if sys.argv[1]=='transfer-extra':
 payload={n:base64.b64encode((D/n).read_bytes()).decode() for n in ['launch.py','invoke.py','LAUNCH_MANIFEST.json']}
 code="from pathlib import Path\nimport base64,json\nD=Path(%r)/'incoming/Kimi-operator-search-r339'\npayload=%r\nfor n,v in payload.items():\n raw=base64.b64decode(v)\n if (D/n).exists():assert (D/n).read_bytes()==raw,n\n else:(D/n).write_bytes(raw)\nprint(json.dumps({'extra_transferred':len(payload)}))"%(R,payload)
elif sys.argv[1]=='launch':
 commit=sys.argv[2];assert len(commit)==40
 code="import os,runpy\nos.environ['SOURCE_COMMIT']=%r\nrunpy.run_path(%r,run_name='__main__')"%(commit,R+'/incoming/Kimi-operator-search-r339/launch.py')
elif sys.argv[1]=='results':
 code="""from pathlib import Path
import json,subprocess
R=Path(%r);D=R/'incoming/Kimi-operator-search-r339'
print(json.dumps({'results':{p.name:json.loads(p.read_text()) for p in (R/'results').glob('Kimi339-*.json')},'logs':{p.name:p.read_text()[-14000:] for p in D.glob('GPU-*.log')},'accounting':subprocess.check_output(['sacct','-S','2026-11-01','-j','910371','-n','-P','-o','JobID,JobName%%50,State,ExitCode,Elapsed,NodeList'],text=True)[-16000:]}))
"""%R
else:raise ValueError(sys.argv[1])
print(json.dumps(remote(code),ensure_ascii=False,indent=2))

