"""Fetch only the fixed69 original Python source files; no models or samples."""
from pathlib import Path
import base64,hashlib,json,subprocess
D=Path(__file__).resolve().parents[1];B=D.parent;G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test');A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline');claim=json.loads((D/'source-publication-receipt.json').read_bytes())['commit'];git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)];mf=json.loads(subprocess.check_output(git+['show',claim+':code/bw1000/training-vggt-native-provider-HIP-CPU-v1/UPSTREAM_FILES.json']));assert len(mf['files'])==69 and sum(p['bytes'] for p in mf['files'])==552106
code="""from pathlib import Path
import base64,hashlib,json
root=Path(MF['root']);out={}
for p in MF['files']:
 part=Path(p['name']);assert not part.is_absolute() and '..' not in part.parts;f=root/part;assert f.is_file() and not f.is_symlink() and f.stat().st_size==p['bytes'];raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==p['SHA256'];out[p['name']]=base64.b64encode(raw).decode()
print(json.dumps(dict(files=out,source_bytes=sum(p['bytes'] for p in MF['files']),model_weight_bytes=0,raw_input_bytes=0,GPU_cards=0)))
"""
payload=('MF='+repr(mf)+'\n'+code).encode()
with (D/'source-fetch-intent.json').open('x') as f:json.dump(dict(claim=claim,payload_SHA256=hashlib.sha256(payload).hexdigest(),Python_source_files=69,Python_source_bytes=552106,model_weight_bytes=0,raw_input_bytes=0,exec_request_attempted=True,replay_allowed=False),f,indent=2)
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),'a100-26','python3 -B -u -'];r=subprocess.run(ssh,input=payload,capture_output=True,timeout=45);(D/'source-fetch-stderr.private.txt').write_bytes(r.stderr);assert r.returncode==0,r.stderr[-200:];q=json.loads(r.stdout);assert q['source_bytes']==552106 and set(q['files'])=={p['name'] for p in mf['files']}
for pin in mf['files']:
 raw=base64.b64decode(q['files'][pin['name']],validate=True);assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];p=D/'original-upstream'/pin['name'];p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
value=dict(passed=True,claim=claim,Python_source_files=69,Python_source_bytes=552106,stdout_SHA256=hashlib.sha256(r.stdout).hexdigest(),model_weight_bytes=0,raw_input_bytes=0,GPU_cards=0);(D/'source-fetch-outcome.json').write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value))
