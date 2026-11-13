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
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-single-drain-TP16-admission-v2')
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
assert len(accounting)==1 and set(query(['scontrol','show','hostnames',accounting[0][5]]).splitlines())==set(i['actual_nodes'])
assert digest(w/'CPU.log')==o['log_SHA256']
for name,h in i['source_files'].items():
    assert digest(w/'src'/name)==h
b=o.get('build')
if b is not None:
    assert b==parse(w/'build-result.json') and digest(w/'build-result.json')==o['build_SHA256']
if o['passed']:
    assert b['passed'] and o['GPU_cards']==0 and accounting[0][2:4]==['COMPLETED','0:0']
    binding=parse(w/'src/BINDING.json')
    assert binding['nodes']==i['actual_nodes'] and len(b['ranks'])==2
    for rank,(row,pin) in enumerate(zip(b['ranks'],b['rank_SHA256'],strict=True)):
        file=w/f'rank-{rank}.json'
        assert digest(file)==pin and row==parse(file)
        assert row['node']==binding['nodes'][rank] and row['step']==accounting[0][0] and row['rank']==rank
        assert all(row[key] is True for key in ('passed','model_metadata_verified','tokenizer_verified','TCP_both_directions'))
        assert row['libraries_verified']==2 and row['GPU_cards']==0 and row['model_weights_read'] is False
        assert row['index_keys']>=117060 and row['shard_count']>0 and row['shard_bytes']>0
        assert len(set(row['addresses']))==2
        assert parse(w/f'listener-{rank}.json')==dict(step=row['step'],node=row['node'],addresses=row['addresses'])
    assert all(b['ranks'][0][key]==b['ranks'][1][key] for key in ('addresses','shard_count','shard_bytes','index_keys','shard_sizes_SHA256'))
print(json.dumps(dict(outcome_exists=True,verified=True,passed=o['passed'],build=b,accounting=accounting,
                      outcome_SHA256=digest(w/'outcome.json'),GPU_cards=0,
                      failure_types=[] if o['passed'] else sorted({line.split(':',1)[0].strip() for line in (w/'CPU.log').read_text(errors='replace').splitlines() if re.match(r'^(AssertionError|FileNotFoundError|OSError|TimeoutError|PermissionError|socket[.]gaierror):?',line)}))))
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
print(json.dumps(dict(verified=v.get("verified"), passed=v.get("passed"), accounting=v.get("accounting"), active=v.get("active"), outcome_exists=v.get("outcome_exists"), failure=(v.get("build") or {}).get("detail"), rc=v.get("rc"), failure_types=v.get("failure_types"), ranks=[{k:r[k] for k in ("node","index_keys","shard_count","shard_bytes")} for r in (v.get("build") or {}).get("ranks",[])])))
