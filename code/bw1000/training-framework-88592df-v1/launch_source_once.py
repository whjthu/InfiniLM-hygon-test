"""Stage only the published full framework archive through existing SSH, once."""
import base64,hashlib,json,shlex,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;G=Path(sys.argv[1]);claim=sys.argv[2]
def blob(name):return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',claim+':'+name])
task=json.loads(blob('coordination/tasks/training-framework-88592df-v1.json'));sources={}
for name,pin in task['source_SHA256'].items():
    data=blob(task['source']+'/'+name);assert hashlib.sha256(data).hexdigest()==pin,name;sources[name]=data
bound={'verify_entry.py':{'b64':base64.b64encode(sources['verify_entry.py']).decode(),'SHA256':task['source_SHA256']['verify_entry.py']}}
code=f"CLAIM={claim!r}\nEXPECTED_NODES={task['expected_nodes']!r}\nMANIFEST_B64={base64.b64encode(sources['FRAMEWORK_FILES.json']).decode()!r}\nMANIFEST_SHA={task['source_SHA256']['FRAMEWORK_FILES.json']!r}\nHELPERS={bound!r}\n"+sources['receive_source_once.py'].decode()
header=(json.dumps({'code_b64':base64.b64encode(code.encode()).decode()})+'\n').encode();assert len(header)<131072
bootstrap="import base64,json,sys;line=sys.stdin.buffer.readline(131073);assert len(line)<=131072 and line.endswith(b'\\n');exec(compile(base64.b64decode(json.loads(line)['code_b64']),'<frozen_source_receiver>','exec'))"
ns={'__file__':str(Path(__file__).resolve())};prelude=(BASE.parent/'i03-deep-c4-history-staging-v1/read_source_once.py').read_text(encoding='utf8')
exec(compile(prelude.split('REMOTE=',1)[0],'existing_cluster_ssh','exec'),ns)
ssh=ns['ssh'].copy();ssh[-1]='/usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(bootstrap)
with (BASE/'launch-intent.json').open('x') as f:json.dump({'claim_commit':claim,'archive_SHA256':task['source_SHA256']['FRAMEWORK.tar.gz']},f)
r=subprocess.run(ssh,input=header+sources['FRAMEWORK.tar.gz'],capture_output=True,timeout=130)
with (BASE/'original.stdout').open('xb') as f:f.write(r.stdout)
with (BASE/'original.stderr').open('xb') as f:f.write(r.stderr)
out={'transport_rc':r.returncode,'stdout_SHA256':hashlib.sha256(r.stdout).hexdigest(),'stderr_SHA256':hashlib.sha256(r.stderr).hexdigest()}
if r.returncode==0:out['value']=json.loads(r.stdout)
with (BASE/'outcome.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
