from pathlib import Path
import json
D=Path(__file__).resolve().parent;g=json.loads((D/'private-GO-r1-prepublish.json').read_bytes());g['publication_commit']='48f2b4df97e764df5d5708d355555923b3f83bb1'
claim=json.loads((D.parents[2]/'results/bw1000/2026-11-02/GLM361-r1-source-CPU-resource-claim.json').read_bytes())
for k in ['baseline_355_root_audit_path','baseline_355_root_audit_SHA256','baseline_355_root_audit_publication']:g[k]=claim[k]
remote='''from pathlib import Path
import json,os,sys,subprocess
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-prefill-window-interleave-r361/runtime-r1/controller';g=GO
go=R/'results/GLM-r361-execution-GO-48f2b4df.json';assert not go.exists()
with go.open('x')as f:json.dump(g,f,indent=2);f.flush();os.fsync(f.fileno())
p=subprocess.run([sys.executable,'-B',str(D/'invoke.py'),'--execution-go',str(go)],capture_output=True,text=True,timeout=150)
if p.returncode:print(json.dumps({'launcher_failed':True,'returncode':p.returncode,'stdout':p.stdout[-4000:],'stderr':p.stderr[-5000:],'no_replay':True}))
else:
 q=json.loads(p.stdout.strip().splitlines()[-1]);q.update(execution_GO_path=str(go),publication_commit=g['publication_commit'],owner_manifest_SHA256=g['owner_manifest_SHA256'],controller_manifest_SHA256=g['controller_manifest_SHA256']);print(json.dumps(q))
'''.replace('g=GO','g='+repr(g));compile(remote,'remote-launch-once-r1','exec');(D/'remote_launch_r1.py').write_bytes(remote.encode())
