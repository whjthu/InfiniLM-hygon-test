"""Exercise the new real stager prefix locally, with only namespace/owner mocked."""
from pathlib import Path
import ast,base64,contextlib,copy,gzip,hashlib,io,json,os,time
D=Path(__file__).resolve().parent;S=D/'publication';source=S/'code/bw1000'/D.name
task=json.loads((S/('coordination/tasks/'+D.name+'.json')).read_bytes())
files={n:dict(b64=base64.b64encode((source/n).read_bytes()).decode(),bytes=(source/n).stat().st_size,SHA256=task['source_SHA256'][n]) for n in task['remote_files']}
code=(D/'receive_reference.py').read_text(encoding='utf8').split("env=os.environ.copy()",1)[0]
assert 'subprocess.run' not in code and 'from torch' not in code
owner="owner=dict(host=socket.gethostname(),PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))"
assert owner in code
fixtures=D/'local-staging-fixtures';fixtures.mkdir(exist_ok=False);results=[]
def run(label,values,success):
 target=(fixtures/label).resolve();assert fixtures.resolve() in target.parents and not target.exists()
 text=code.replace("Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1')",'Path('+repr(str(target))+')').replace(owner,"owner=dict(host='local-CPU-fixture',PID=os.getpid(),PGID=0,startticks=0)")
 ast.parse(text);out=io.StringIO();started=time.monotonic();error=None
 try:
  with contextlib.redirect_stdout(out):exec(compile(text,'real_stager_local_fixture','exec'),dict(CLAIM='1'*40,REQUEST='local-'+label,FILES=values))
 except BaseException as e:error=type(e).__name__
 assert (error is None)==success,(label,error)
 count=0
 if success:
  manifest=json.loads((source/'FRAMEWORK_FILES.json').read_bytes())
  for name,pin in manifest['files'].items():
   raw=(target/'framework'/name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'];count+=1
  assert count==349 and (target/'installed.json').is_file()
 results.append(dict(case=label,passed=True,expected_success=success,actual_error_type=error,framework_files_verified=count,
  wall_seconds=time.monotonic()-started,namespace_redirected=True,owner_proc_api_mocked=True,actual_remote_install=False,GPU_cards=0,stdout_SHA256=hashlib.sha256(out.getvalue().encode()).hexdigest()))
run('complete349',files,True)
bad=copy.deepcopy(files);bad['FRAMEWORK_BUNDLE.json.gz']['SHA256']='0'*64;run('wrong-bundle-SHA',bad,False)
bad=copy.deepcopy(files);raw=b'outside marker must not be written';bad['../outside-marker.txt']=dict(b64=base64.b64encode(raw).decode(),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
assert not (fixtures/'outside-marker.txt').exists();run('path-traversal',bad,False);assert not (fixtures/'outside-marker.txt').exists()
bad=copy.deepcopy(files);raw=gzip.compress(b'x'*(16*1024**2+1),mtime=0);bad['FRAMEWORK_BUNDLE.json.gz']=dict(b64=base64.b64encode(raw).decode(),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());run('oversized-expanded-bundle',bad,False)
v=dict(passed=True,actual_stager_prefix_SHA256=hashlib.sha256(code.encode()).hexdigest(),cases=results,scope='Local CPU stager/file-boundary fixtures only; namespace and Linux owner API redirected. No targetHOST,model,GPU or actual remote-install qualification.')
with (D/'staging-CPU-result.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(dict(passed=True,cases=len(results),complete_framework_files=349,actual_remote_install=False,GPU_cards=0)))
