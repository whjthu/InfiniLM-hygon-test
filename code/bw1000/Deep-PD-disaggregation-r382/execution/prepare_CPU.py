"""Freeze the private execution bundle and one actual Linux CPU claim, never GPU."""
from pathlib import Path
import ast,json,hashlib,base64,subprocess,sys

D=Path(__file__).resolve().parent;G=D.parents[3];L=G.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=json.loads((D/'NATIVE_PIN.json').read_bytes());assert pin['ready']and pin['CPU_receipt_SHA256']=='60b7d4801bec2f826984055c6c1ee1d1bcd15ef1235a3559f23b268f7a574499'
for p in D.rglob('*.py'):ast.parse(p.read_bytes())
deps=json.loads((D/'DEPENDENCIES.json').read_bytes());assert all(sha(D/'root_support'/(name+'.py'))==h for name,h in deps['root_modules_SHA256'].items())
assert sha(D/'root_support/BENCHMARK_PLAN.json')==deps['plan_SHA256']=='ff0902853e1d1eb78b1cbb732f3d9e1b70f36581c4f970485dbd9de279366680'
files=sorted(str(p.relative_to(D)).replace('\\','/')for p in D.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.name not in ['SOURCE_MANIFEST.json','GPU_MANIFEST.json'])
claim=dict(parent=910407,node='a14r2n20',CPU_cores=4,host_mem_MiB=8192,hard_s=600,inside_s=570,GPU_cards=0,Model_loads=0,Native_dlopen=False,one_attempt=True)
manifest=dict(schema=1,GPU_GO=False,scope='Actual private execution entry CPU qualification: real52object derive+actual8489fixture once, actualcontrol/socket/stages/codec/wire/client with model/parameter/owner recording leaves; no GPU orNative dlopen.',files_SHA256={n:sha(D/n)for n in files},dependencies=deps,Native_CPU_receipt_SHA256=pin['CPU_receipt_SHA256'],CPU_claim=claim,planned_external=15,planned_stages=22)
(D/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');h=sha(D/'SOURCE_MANIFEST.json')
result=G/'results/bw1000/2026-11-02/Deep382-execution-local-wholeentry-CPU-PASS.json'
p=subprocess.run([sys.executable,'-B',str(D/'check_entry_CPU.py')],capture_output=True,text=True,timeout=60);assert p.returncode==0,p.stderr
q=json.loads(p.stdout);q.update(source_manifest_SHA256=h,local_only=True,actual_Linux_binding_and_official_encoder='PENDING separate CPU claim',local_first_AST_failure='CPU_leaves missing whitespace in ternary else; repaired before any cluster submission or sourcefreeze; no CPU/GPU Slurm step')
result.write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
task=G/'coordination/tasks/Deep-PD-execution-r382.json'
task.write_text(json.dumps(dict(owner='/root/kimi_operator_tuning',state='SOURCE_LOCAL_CPU_PASS_LINUX_CPU_CLAIM',source=str(D.relative_to(G)).replace('\\','/'),source_manifest_SHA256=h,GPU_GO=False,GPU_submitted=False,Native_dlopen=False,CPU_claim=claim,planned_external=15,planned_stage_lifecycles=22,plan_SHA256=deps['plan_SHA256'],Native_pin=pin,local_entry_result=str(result.relative_to(G)).replace('\\','/'),GPU_future_not_claimed=dict(parent=910376,P='a02r4n12',D='a02r4n13',cards=16,TP_per_role=8,PP=1,rows=1,capacity=32768,hard_s=3600,CPU_keeper_s=3900),next='One sourcepublished4coreLinuxCPU actualderive/objectgraph/official8489 callback/wholeentry. FinalGPUclaim onlyafter realCPUreceipt and frozen finalmanifest, then freshidle once residency with plannedwarm qualityguard; no separateABI GPU orpromptreplay.',no_status_write=True),indent=2)+'\n',encoding='utf8')
payload={n:base64.b64encode((D/n).read_bytes()).decode()for n in files+['SOURCE_MANIFEST.json']}
remote="from pathlib import Path\nimport json,base64,hashlib\nD=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/Deep-PD-disaggregation-r382/execution');D.mkdir(exist_ok=False)\npayload="+repr(payload)+"\nfor n,b in payload.items():\n p=D/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))\nsha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()\nassert sha(D/'SOURCE_MANIFEST.json')=="+repr(h)+"\nm=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==v for n,v in m['files_SHA256'].items())\nprint(json.dumps(dict(uploaded=True,source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),CPU_submitted=False,GPU=0)))\n"
ast.parse(remote);(L/'remote382-execution-upload.py').write_text(remote,encoding='utf8')
print(json.dumps(dict(source_manifest_SHA256=h,files=len(files),local_cases=len(q['cases']),CPU_claim=claim,GPU=0)))
