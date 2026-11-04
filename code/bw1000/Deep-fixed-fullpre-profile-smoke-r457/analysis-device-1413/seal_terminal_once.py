"""Terminal-only immutable R/L raw archive. No requests/jobs/control/polling."""
from pathlib import Path
import base64,datetime,gzip,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label='Deep-r457-fixed-profile-smoke-fixed-profile-20261105-a';folder=R/'results/Deep-fixed-profile-public-r457/fixed-profile-20261105-a';client=R/'results'/(label+'-client');watch=R/'results'/(label+'-watch.json')
w=json.loads(watch.read_bytes());assert w['completed'],'Not terminal; do not wait/poll here'
assert w['actual_GPU']=='910374.1413'and w['CPU_keeper']=='910407.965'and w['original_GPU_start_unix']==1793766297.782444
archive=R/'results/Deep457-1413-terminal-sealed-v1.json.gz';assert not archive.exists(),'Consume original archive; no reseal/replay'
provider=R/'incoming/Deep-fixed-fullpre-experimental-profile-r454/source-v1';deploy=R/'incoming/Deep-fixed-fullpre-profile-smoke-r457/deployment-gpu-v1'
paths=[watch,R/'results'/(label+'-intent.json'),R/'results'/(label+'-preparation.json'),*folder.glob('*.json'),*client.glob('*'),R/'results'/(label+'-client-identity.json'),provider/'SOURCE_MANIFEST.json',deploy/'SOURCE_MANIFEST.json',deploy/'MODEL_EXECUTION_CLAIM.json',R/'candidates/Deep454-fixed-profile-CPU-v1/result.json',R/'candidates/Deep457-smoke-glue-CPU-v1/result.json']
for base in[provider,deploy]:
 for name in json.loads((base/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256']:paths.append(base/name)
card=json.loads((deploy/'FIXTURE_CARD.json').read_bytes())
paths.extend(Path(card[key+'_path'])for key in['cases','requests','encoding'])
model=Path('/public/home/aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');paths.extend([model/'config.json',model/'generation_config.json'])
files={}
for path in sorted(set(paths)):
 if not path.is_file():continue
 raw=path.read_bytes();files[str(path)]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),encoding='base64',base64=base64.b64encode(raw).decode())
command=['sacct','-S','2026-10-31','-j','910374,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,ElapsedRaw,NodeList%100'];a=subprocess.run(command,capture_output=True,text=True,timeout=30);assert a.returncode==0
rows=[line.split('|')for line in a.stdout.splitlines()if label in line]
q=dict(schema='Deep457-original-terminal-only-v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),run='fixed-profile-20261105-a',files=files,accounting=rows,accounting_argv=command,raw_weights_or_activations=False,new_requests=0,new_jobs=0,original_deadlines_preserved=True)
raw=(json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n').encode();compressed=gzip.compress(raw,compresslevel=6,mtime=0)
with archive.open('xb')as out:out.write(compressed)
print(json.dumps(dict(path=str(archive),bytes=len(compressed),SHA256=hashlib.sha256(compressed).hexdigest(),uncompressed_bytes=len(raw),uncompressed_SHA256=hashlib.sha256(raw).hexdigest(),files=len(files),accounting=rows)))
