from pathlib import Path
from hashlib import sha256
import json,socket,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/GLM952-prefix-continuation/gpu-continuation-v5'
expected=EXPECTED
assert not D.exists()
assert sha256(files['GPU_MANIFEST.json']).hexdigest()==expected
m=json.loads(files['GPU_MANIFEST.json'])
assert set(files)==set(m['files_SHA256'])|{'GPU_MANIFEST.json'}
assert all(sha256(files[n]).hexdigest()==h for n,h in m['files_SHA256'].items())
D.mkdir()
for name,b in files.items():
    assert Path(name).name==name
    with (D/name).open('xb') as stream:stream.write(b)
p=subprocess.run(['python3','-B',str(D/'launch_once.py'),'--manifest-sha',expected,
    '--publication-commit',PUBLICATION],capture_output=True,text=True,timeout=90)
print(json.dumps(dict(HOST=socket.gethostname(),publication=PUBLICATION,manifest_SHA256=expected,
    rc=p.returncode,result=json.loads(p.stdout) if p.returncode==0 else None,stderr_tail=p.stderr.splitlines()[-10:])))
