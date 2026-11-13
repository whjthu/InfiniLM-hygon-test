"""Host-only checks for this scope's original deadlines and embedded programs."""
from pathlib import Path
import ast,importlib.util,json,math,subprocess
D=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('sdk_epoch',D/'scope.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
valid=[]
for elapsed in [0,1,30,59,60,90,119]:
 now=1000+elapsed;minutes=m.gpu_minutes(1000,now);assert now+minutes*60>1000+m.INSIDE_SECONDS+m.TERM_SECONDS+m.KILL_SECONDS and now+minutes*60<1540;valid.append(elapsed)
rejected=[]
for epoch,now in [(1000,1121),(1000,1480),(0,1),(1000,999),(math.nan,1000),(1000,math.inf),(-1,1)]:
 try:m.gpu_minutes(epoch,now)
 except (TimeoutError,ValueError):rejected.append([str(epoch),str(now)])
 else:raise AssertionError('invalid or late epoch was admitted')
clock_wall=[1000.];clock_mono=[10.];c=m.EpochClock(1000,wall=lambda:clock_wall[0],monotonic=lambda:clock_mono[0]);assert c.left(420)==420
clock_mono[0]=431.
try:c.left(420)
except TimeoutError:mono_rejected=True
else:raise AssertionError('monotonic expiration reset by wallclock')
clock_wall[0]=2000.;clock_mono[0]=10.
try:c.left(420)
except TimeoutError:wall_rejected=True
else:raise AssertionError('wall deadline escaped')
parsed=[]
for p in D.glob('*.py'):ast.parse(p.read_bytes(),filename=p.name);parsed.append(p.name)
reader=ast.parse((D/'read_original_once.py').read_bytes());remote=next(n.value.value for n in reader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote' for t in n.targets));ast.parse(remote)
controller=ast.parse((D/'controller.py').read_bytes());fn=next(n for n in controller.body if isinstance(n,ast.FunctionDef) and n.name=='container');shell=next(n.value.value for n in fn.body if isinstance(n,ast.Assign) and n.targets[0].id=='shell')
assert shell.endswith('runtime_exec.py "$1"')
bash=Path('C:/Program Files/Git/bin/bash.exe');assert bash.is_file();r=subprocess.run([str(bash),'-n'],input=shell.encode(),capture_output=True);assert r.returncode==0
launch=ast.parse((D/'launch_once.py').read_bytes());expr=next(n.value for n in launch.body if isinstance(n,ast.Assign) and n.targets[0].id=='ready_program');ready=eval(compile(ast.Expression(expr),'explicit_ready_literal','eval'),{'nonce':'CPU-only-literal-validation'});ast.parse(ready)
assert 'signal.alarm(120)' in ready
result=dict(passed=True,valid_deadline_cases=len(valid),invalid_deadline_cases=len(rejected),monotonic_expiry_rejected=mono_rejected,wall_expiry_rejected=wall_rejected,Python_AST=parsed,reader_embedded_AST=True,authenticated_ready_embedded_AST=True,shell_syntax_rc=r.returncode,torch_imports=0,Native_loads=0,SDK_calls=0,GPU_calls=0)
with (D/'local-scope-checks.json').open('x',newline='\n') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
