"""ONE bounded readonly metadata/source SSH; no key/env/job/weight access."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).parent
remote=r'''
from pathlib import Path
import hashlib,json,shlex
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
def path(s):return Path(str(s).replace('/refactor',str(root),1))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cpu=root/'candidates/deep-c4-smallt-model-r9254/CPU-build.json'
assert digest(cpu)=='43d4478102462e26bbfb0784942ec2f8f05228f53fb42eae0b23c051d1ed6d4c'
c=json.loads(cpu.read_bytes());objects=c['link']['full']['objects']
hist=root/'results/deep-c4-projection-model-r5920-build.json';h=json.loads(hist.read_bytes())
out={'CPU_receipt_SHA256':digest(cpu),'current_link':c['link']['full'],'historical_build_receipt_SHA256':digest(hist),'units':[],'files':{},'MD':[]}
names={'deepseek_v4_temporal_compressor.hpp','deepseek_v4_paged_latent_cache.hpp','deepseek_v4_c4_indexer.hpp','deepseek_v4_swa_attention.hpp','deepseek_v4_hca_attention.hpp','deepseek_v4_csa_attention.hpp','deepseek_v4_native_routed_decoder_r20.hpp','deepseek_v4_routed_text_model_r20.hpp'}
for u in h['units']:
 obj=u['object'];current=objects.get(obj)
 if current and current==u['object_sha256']:
  deps=u['dependencies'];matched=[d for d in deps if Path(d['path']).name in names]
  if matched:
   record={'source':u['source'],'source_SHA256':u['source_sha256'],'object':obj,'object_SHA256':current,'dependencies':matched}
   out['units'].append(record)
   # Named actual cache-owning HOST source; no arbitrary dependency execution.
   p=path(u['source']);out['files'][u['source']]={'SHA256':digest(p),'text':p.read_text()}
   for d in matched:
    p=path(d['path']);out['files'][d['path']]={'SHA256':digest(p),'expected_from_build':d['sha256'],'text':p.read_text()}
# Current overridden attention/decoder dependency metadata, bounded to current
# object directories and named .d files; list then read metadata only.
parents=sorted({str(path(o).parent) for o in objects})
for directory in parents:
 p=Path(directory)
 candidates=list(p.glob('*.d'))+list((p/'deps').glob('*.d'))
 for dep in sorted(set(candidates))[:80]:
  text=dep.read_text()
  if not any(n in text for n in names):continue
  words=shlex.split(text.replace('\\\n',' '));target=words[0].rstrip(':')
  logical=target.replace(str(root),'/refactor',1)
  if logical not in objects:continue
  selected=[x for x in words[1:] if Path(x).name in names]
  out['MD'].append({'path':str(dep),'SHA256':digest(dep),'object':logical,'headers':selected})
  for source in selected:
   f=path(source)
   if f.is_file():out['files'][source]={'SHA256':digest(f),'text':f.read_text()}
print(json.dumps(out))
'''
args=['ssh','-T','-p','65032','-i','C:/Users/WangHaojie/.ssh/hygon-261012.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile=C:/Users/WangHaojie/Documents/ChatGPT/海光集群/known_hosts','aclsylqidf@zzeshell.scnet.cn','python3 -']
r=subprocess.run(args,input=remote,text=True,capture_output=True,timeout=100)
if r.returncode:raise RuntimeError(r.stderr[:2500])
data=json.loads(r.stdout);dest=ROOT/'actual_binding_readonly_r124.json';dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='READONLY_BINDING_RETAINED',units=len(data['units']),files=len(data['files']),MD=len(data['MD']),path=str(dest))))
