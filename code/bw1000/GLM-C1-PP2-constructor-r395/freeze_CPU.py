from pathlib import Path
import ast,base64,hashlib,json
D=Path(__file__).resolve().parent;G=D.parents[2];L=G.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parents=[G/'code/bw1000/GLM-W4-cross-workload-r349/owner/native_chain_r186.py',
         G/'code/bw1000/GLM-W4-cross-workload-r349/owner/SOURCE_MANIFEST.json',
         G/'code/bw1000/agent-context-32k/GLM-PP4-capacity-private-r186/allocation_entry_r186.py',
         G/'code/bw1000/agent-context-32k/GLM-PP4-capacity-private-r186/GPU_inside_r186.sh',
         G/'code/bw1000/glm-pp3-constructor/glm_PP3_constructor_r8781.py',
         G/'analysis/bw1000/GLM-C1-PP2-current-capacity-feasibility-r394.json']
manifest=dict(files_SHA256={p.name:sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name!='SOURCE_MANIFEST.json'},
              CPU_claim=json.loads((D/'DESIGN.json').read_bytes())['CPU_claim'],GPU_GO=False,Model_GO=False,
              Native_SHA256='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00',
              parent_sources=[dict(path=p.relative_to(G).as_posix(),SHA256=sha(p)) for p in parents])
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n');h=sha(D/'SOURCE_MANIFEST.json')
payload={p.name:base64.b64encode(p.read_bytes()).decode() for p in sorted(D.iterdir()) if p.is_file()}
script='''from pathlib import Path
import ast,base64,hashlib,json,os,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-C1-PP2-constructor-r395';MF=__MF__
label='GLM-C1-PP2-r395-CPU'
rows=subprocess.check_output(['sacct','-S','2026-10-31','-j','910407','-n','-P','-o','JobIDRaw,JobName%160,State'],text=True,timeout=25)
assert not any(x.split('|')[1]==label for x in rows.splitlines() if '|'in x),'Existinglabel: follow original, never replay'
assert not D.exists() and not (R/'candidates/GLM-C1-PP2-constructor-r395').exists()
D.mkdir()
for name,value in __PAYLOAD__.items():(D/name).write_bytes(base64.b64decode(value))
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==MF
for p in D.glob('*.py'):ast.parse(p.read_text())
with(R/'results/GLM-C1-PP2-r395-CPU-intent.json').open('x')as f:json.dump(dict(source_manifest_SHA256=MF,GPU=0,retry=False),f)
env=dict(os.environ,WAVE_INSIDE_R2323=D.name+'/CPU_inside.sh',GLM_R395_CPU_SOURCE_SHA=MF)
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:05:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'--export=ALL','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(R/'logs'/(label+'-client.log')).open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(CPU_submitted=True,CPU_only=True,GPU=0,Model_loads=0,compile=False,PID=p.pid,label=label,source_manifest_SHA256=MF)))
'''
script=script.replace('__MF__',repr(h)).replace('__PAYLOAD__',repr(payload));ast.parse(script)
(L/'remote395-upload-CPU.py').write_text(script,encoding='utf8',newline='\n')
print(json.dumps(dict(source_manifest_SHA256=h,files=len(manifest['files_SHA256']),helper_AST=True,GPU_GO=False)))
