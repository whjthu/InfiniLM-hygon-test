"""Read and seal existing terminal artifacts only. Never submit/probe/connect."""
from pathlib import Path
import datetime,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-NoV-16card-W4-pair-r414';RUN='pair-20261103-a';LABEL='GLM-NoV-16card-W4-pair-r414-'+RUN;GPU='910376.853';KEEPER='910407.833';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dest=R/'results/GLM414-853-original-terminal-sealed-archive-R-only.json'
if dest.exists():
 a=json.loads(dest.read_bytes());assert a['actual_GPU']==GPU and a['actual_keeper']==KEEPER;print(json.dumps(dict(archive_path=str(dest),archive_SHA256=sha(dest),archive_bytes=dest.stat().st_size,archive=a,existing_sealed=True)));raise SystemExit(0)
watch=R/'results'/(LABEL+'-watch.json');w=json.loads(watch.read_bytes());assert w['completed']and w['actual_GPU']==GPU and w['CPU_keeper']==KEEPER
fields=['JobIDRaw','JobName','State','ExitCode','Elapsed','ElapsedRaw','Start','End','NodeList']
text=subprocess.check_output(['sacct','-S','2026-10-31','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,Elapsed,ElapsedRaw,Start,End,NodeList%100'],text=True,timeout=30)
accounting=[dict(zip(fields,line.split('|')))for line in text.splitlines()if LABEL in line];byid={r['JobIDRaw']:r for r in accounting}
assert GPU in byid and KEEPER in byid
files={};roles={};missing=[]
def host(p):
 p=Path(p)
 if p.is_relative_to('/refactor'):return R/p.relative_to('/refactor')
 return p
def add(role,path):
 p=host(path);key=str(p)
 if not p.is_file():missing.append(dict(role=role,path=key));return None
 roles[role]=key
 if key not in files:
  raw=p.read_bytes();assert len(raw)<64*1024*1024,(key,len(raw));s=raw.decode('utf-8');assert s.encode()==raw
  files[key]=dict(path=key,SHA256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),encoding='utf-8',rawtext=s)
 return json.loads(files[key]['rawtext'])if p.suffix=='.json'else None
add('watch',watch);add('intent',R/'results/GLM-r414-Model-intent.json');add('runtime_GO',R/'control'/(LABEL+'-GO.json'))
for stage in range(2):
 for rank in range(8):
  tag='s%d-r%d'%(stage,rank);p=R/'results'/('GLM-latest-full-owner-r290-'+RUN+'-stage%d-rank%d.json'%(stage,rank));q=add('owner:'+tag,p)
  if q is None:continue
  assert q['actual_producer']==GPU
  add('lifecycle:'+tag,R/'results'/('GLM-r404-lifecycle-'+RUN+'-stage%d-rank%d.json'%(stage,rank)))
  snap=host(q['constructor_snapshot_path']);assert sha(snap)==q['constructor_snapshot_SHA256'];add('constructor:'+tag,snap)
  add('phase:'+tag,R/'results'/('GLM-Model-phase-r9183-'+RUN+'-stage%d-rank%d.jsonl'%(stage,rank)))
add('supervisor',R/'results'/('GLM-latest-full-owner-r290-supervisor-'+RUN+'.json'))
add('ready',R/'results'/('GLM-latest-full-owner-r290-ready-'+RUN+'.json'))
client=R/'results'/('GLM-latest-full-owner-r290-client-'+RUN)/'paired';add('client_result',client/'result.json')
for i in range(6):add('response:'+str(i),client/('response-%02d.wire'%i))
add('plan',S/'PLAN.json');m=add('GPU_manifest',S/'GPU_MANIFEST.json');assert sha(S/'GPU_MANIFEST.json')=='3a9cbe1e68706369ba8bdd319ec574d52812e007e20dc7a0f9008a58c400fdd8'
owner=add('served_owner_manifest',S/'owner/SOURCE_MANIFEST.json');assert sha(S/'owner/SOURCE_MANIFEST.json')==m['owner_manifest_SHA256']
for name,h in owner['files_SHA256'].items():assert sha(S/'owner'/name)==h;add('served_source:'+name,S/'owner'/name)
source=add('root_source_manifest',S/'SOURCE_MANIFEST.json');assert sha(S/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256']
for name,h in source['files_SHA256'].items():assert sha(S/name)==h;add('ancestry_source:'+name,S/name)
for name,h in m['additional_files_SHA256'].items():assert sha(S/name)==h;add('entry_source:'+name,S/name)
for name in ['collect.py','controller/idle.py','controller/resource_wave_host_alias_outer.sh']:add('lifecycle_source:'+name,S/name)
for role,path in [('Native_CPU804','/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json'),('CPU832','/refactor/candidates/GLM-NoV-16card-W4-pair-r414-CPU/CPU-result.json'),('CPU820','/refactor/candidates/GLM-V-on-demand-Model-r404-CPU-layout-v4/CPU-result.json'),('final_Bash_CLI','/refactor/results/GLM-r414-final-CLI-preflight.json')]:add(role,path)
card=add('fixture_card',S/'owner/FIXTURE_CARD.json')
for i in range(6):
 f=card['fixtures'][0]
 for role,key in [('request','request_external_path'),('original_reply','original_reply_external_path'),('prompt_IDs','current_prepared_prompt_IDs')]:add(role+':'+str(i),f[key])
old=Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3')
for name in ['config.json','generation_config.json','model.safetensors.index.json']:add('model_metadata:'+name,old/name)
# Include all other same-run immutable reports/client files/control/logs. Files
# are not interpreted as executable code, and no other run is traversed.
for where in ['results','control','logs']:
 for p in (R/where).glob('*'+RUN+'*'):
  if p.is_file():add('same_run:'+where+'/'+p.name,p)
  elif p.is_dir():
   for f in p.rglob('*'):
    if f.is_file():add('same_run:'+str(f.relative_to(R)),f)
library=host(json.loads((host('/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json')).read_bytes())['library']);assert sha(library)==m['Native_SHA256']
a=dict(schema=1,run=RUN,actual_GPU=GPU,actual_keeper=KEEPER,publication_commit=w['publication_commit'],GPU_manifest_SHA256=sha(S/'GPU_MANIFEST.json'),sealed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting=accounting,roles=roles,files=files,binary_references=[dict(path=str(library),SHA256=sha(library),bytes=library.stat().st_size,raw_bytes_not_copied=True)],file_count=len(files),role_count=len(roles),missing_roles=missing,raw_file_bytes=sum(v['bytes']for v in files.values()),scope='Existing completed same-residency16owner C1TP8PP2/full32768 noV Model;6samecomplete9741 tools windows[2,4,2,4,4,2] ordinary4096, allrawtext retained R/L only. No GPU/CPUjob/probe/Model/HTTPrequest created by archive.',Git_raw_publication=False)
with dest.open('x',encoding='utf-8')as out:out.write(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(archive_path=str(dest),archive_SHA256=sha(dest),archive_bytes=dest.stat().st_size,archive=a,existing_sealed=False)))
