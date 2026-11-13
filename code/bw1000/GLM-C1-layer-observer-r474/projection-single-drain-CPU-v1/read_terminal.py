from pathlib import Path
import ast,hashlib,json,subprocess,sys
D=Path(__file__).resolve().parent
B=D.parent
tree=ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
env={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'known-readonly-transport','exec'),env)
intent=json.loads((D/'launch-intent.json').read_bytes())
remote=r'''
from pathlib import Path
import hashlib,json,subprocess,re
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-projection-single-drain-CPU-v1')
query=lambda argv:subprocess.check_output(argv,text=True,timeout=10)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parse=lambda p:json.loads(p.read_bytes())
accounting=[line.split('|') for line in query(['sacct','-n','-P','-j','1103606','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']).splitlines() if '|'+w.name+'|' in line]
active=[line for line in query(['squeue','--steps','-h','-j','1103606','-o','%i|%j']).splitlines() if w.name in line]
if not (w/'outcome.json').exists():
    print(json.dumps(dict(outcome_exists=False,accounting=accounting,active=active)))
    raise SystemExit(0)
i=parse(w/'intent.json'); o=parse(w/'outcome.json')
assert i['request_id']==o['request_id']==REQUEST and i['claim']==o['claim']==CLAIM
assert accounting==o['accounting'] and not active and o['owned_srun_reaped'] and o['terminal']['terminal_visible']
assert len(accounting)==1 and accounting[0][5]==i['node']=='f13r1n05'
assert digest(w/'CPU.log')==o['log_SHA256']
for name,h in i['source_files'].items():
    assert digest(w/'src'/name)==h
b=o.get('build')
if b is not None:
    assert b==parse(w/'build-result.json') and digest(w/'build-result.json')==o['build_SHA256']
    for cmd in b['commands']:
        assert all(digest(w/(cmd['name']+'.'+kind))==cmd[kind+'_SHA256'] for kind in ('stdout','stderr'))
if o['passed']:
    assert b['passed'] and b['new_HOST_TUs']==1 and b['new_GPU_TUs']==0
    assert b['MD']['equal_fresh_M'] and b['MD']['count']==993 and b['reused_objects']==31 and b['providers']==71
    assert digest(w/'libGLM_projection_host_timing.so')==b['library_SHA256']
    assert digest(w/'prepare.o')==b['object_SHA256']
    assert digest(w/'prepare.before.d')==b['fresh_M']['SHA256'] and digest(w/'prepare.actual.d')==b['MD']['SHA256']
    assert {'modern_glm_projection_host_timing_begin','modern_glm_projection_host_timing_end'}==set(b['diagnostic_exports'])
print(json.dumps(dict(outcome_exists=True,verified=True,passed=o['passed'],build=b,accounting=accounting,
                      outcome_SHA256=digest(w/'outcome.json'),GPU_cards=0)))
'''
args=f"REQUEST={intent['request_id']!r}\nCLAIM={intent['claim']!r}\n"
r=subprocess.run(env['ssh'],input=(args+remote).encode(),capture_output=True,timeout=40)
if r.returncode==0:
    v=json.loads(r.stdout)
else:
    v=dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),
        failure_types=[x for x in ('AssertionError','Connection closed','Permission denied','KeyError') if x.encode() in r.stderr])
tag=sys.argv[1]
assert tag in ('first','second','third')
with (D/(tag+'-independent.json')).open('x') as f:json.dump(v,f,indent=2)
print(json.dumps(dict(verified=v.get("verified"), passed=v.get("passed"), accounting=v.get("accounting"), active=v.get("active"), outcome_exists=v.get("outcome_exists"), failure=(v.get("build") or {}).get("detail"), rc=v.get("rc"), failure_types=v.get("failure_types"))))
