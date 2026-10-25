import pathlib,json,hashlib,ast,difflib,shutil
r=pathlib.Path.cwd();p=r/'Kimi-attnres-front-private-r8982';repo=r/'InfiniLM-hygon-test';dst=repo/'code/bw1000/kimi-attnres-front-fusion';dst.mkdir(exist_ok=True)
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
contract=json.loads((p/'contract_r8982.json').read_text());contract['fixture']='One seeded bank and real layer46 small weights; 2 unmeasured warm calls, 8 timed ABBA+BAAB, final restore A; padded output canary'
contract['performance']='Both modes warmed; only 8 measure-phase calls included. Component only; no Model or bottleneck claim.'
contract['diagnostic']=[x for x in contract['diagnostic']if 'mode call counters'not in x]+['phase counters warm[1,1], measure[4,4],restore[1,0]','32 BF16 output canaries before/after']
(p/'contract_r8982.json').write_bytes((json.dumps(contract,indent=2)+'\n').encode())
old=(r/'modern_kimi_attnres_repeat_probe.cpp').read_text();new=(p/'attnres_explicit_r8982.cpp').read_text();(p/'host_vs_explicit_original.diff').write_bytes(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original_explicitweights',tofile='r8982_explicitweights')).encode())
files=[x for x in p.iterdir()if x.is_file()and x.name!='source_manifest_r8982.json']+[r/n for n in ['build_Kimi_attnres_front_r8983.py','Kimi_attnres_front_CPU_r8983_inside.sh','launch_Kimi_attnres_front_CPU_r8983.py','read_Kimi_attnres_sdk_r8982.py','Kimi-attnres-sdk-r8982.json','prepare_Kimi_attnres_front_r8983.py']]
for x in files:
 if x.suffix=='.py':ast.parse(x.read_text())
 shutil.copyfile(x,dst/x.name)
source={'status':'SOURCE_CLAIM_CPU_BUILD_AUTHORIZED','source_SHA256':{x.name:sha(x)for x in files if x.suffix in ['.py','.cpp','.cu','.sh']}}
(r/'Kimi-attnres-front-source-r8983.json').write_bytes((json.dumps(source,indent=2)+'\n').encode());shutil.copyfile(r/'Kimi-attnres-front-source-r8983.json',dst/'Kimi-attnres-front-source-r8983.json')
manifest={'status':'SOURCE_ONLY_NOT_BUILT','CPU_validation':'Python AST + JSON parse; native not yet compiled','files':{x.name:{'bytes':x.stat().st_size,'SHA256':sha(x)}for x in files}}
(p/'source_manifest_r8982.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode());shutil.copyfile(p/'source_manifest_r8982.json',dst/'source_manifest_r8982.json')
task={'work_item':'qualify-kimi-attnres-front-fusion','owner':'bw1000','state':'SOURCE_PREPARED','method':'RMSNorm+scalarprojection fused; explicit BF16 normalized/score; original Softmax/weightedGemm','scope':'Single M16/C24/cap25/H7168 single-card synthetic bank, real small layer46 weights; not Model','authorization':'CPU-only oneGPU-TU+onehost-TU build; conditional single-card finite component after actual pins reviewed','source':'code/bw1000/kimi-attnres-front-fusion/source_manifest_r8982.json'}
(repo/'coordination/tasks/qualify-kimi-attnres-front-fusion.json').write_bytes((json.dumps(task,ensure_ascii=False,indent=2)+'\n').encode())
mp=repo/'manifests/bw1000/recent-source-files.json';m=json.loads(mp.read_text());known={x['path']:x for x in m['files']}
for x in dst.iterdir():
 rel=x.relative_to(repo).as_posix();item={'path':rel,'bytes':x.stat().st_size,'SHA256':sha(x)}
 if rel in known:known[rel].update(item)
 else:m['files'].append(item)
mp.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps({'source_files':len(list(dst.iterdir())),'source_manifest_SHA256':sha(p/'source_manifest_r8982.json'),'source_proof_SHA256':sha(r/'Kimi-attnres-front-source-r8983.json')}))
