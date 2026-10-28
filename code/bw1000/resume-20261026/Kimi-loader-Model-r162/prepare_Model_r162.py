"""Local source freezer only; no model loading, scheduling, Git or secret values."""
from pathlib import Path
import ast,copy,hashlib,json
D=Path(__file__).resolve().parent;V=D.parent;R=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群')
BASE=V/'Kimi-current-model-sparse-profile-launch-private-r136b';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE/'producer-source-r136b.json')=='8d56b5d38e99f7229fd56a4682a17da07cc562b62392a8cc1bee7fa834f863cb'
producer=json.loads((BASE/'producer-source-r136b.json').read_bytes());M=D/'model';M.mkdir(exist_ok=False)
for rel,h in producer['files_SHA256'].items():
 p=BASE/rel;assert sha(p)==h;out=M/rel.replace('r136b','r162');out.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes()
 if p.suffix in('.py','.json','.sh'):raw=raw.replace(b'r136b',b'r162')
 out.write_bytes(raw)
files=json.loads((R/'CPU-candidates-read-root-r9476-a.json').read_bytes())['files'];item=files[1]
assert item['SHA256']=='35453a2a74468d5bb28643d66263ab40467aa00a0aab8dbd767b819075751944';receipt=item['data']
assert receipt['completed']and receipt['CPU_only']and not receipt['GPU']and not receipt['dlopen']and not receipt['actual_GPU_qualified']
assert receipt['native_SHA256']=='798c4bca01cddb0755a91171ef61dfa755329518bb8cfc87640305a24e55a6e1'
(M/'runtime/model-native-receipt-r9282.json').write_text(json.dumps(receipt,indent=2)+'\n')
SOURCE=V/'Kimi-loader-registry-build503-private-r159a'
(M/'runtime/parameter_loader_r9143.py').write_bytes((SOURCE/'parameter_loader_r159.py').read_bytes())
(M/'runtime/loader_work_observer_r158.py').write_bytes((SOURCE/'loader_work_observer_r158.py').read_bytes())
(M/'runtime/model_registry_contract_r162.py').write_bytes((D/'model_registry_contract_r162.py').read_bytes())
# Explicit mode before Runtime construction; preserve original shca/env/source order.
p=M/'runtime/target_inside.sh';text=p.read_text();assert 'set -euo pipefail'in text
text=text.replace('set -euo pipefail','set -euo pipefail\nexport INFINILM_LOADING_BINDINGS_R9212=1\nexport INFINILM_KIMI_W13_WAVEPARTIAL_R155=0',1);p.write_bytes(text.encode())
p=M/'runtime/native_binding_r9143.py';text=p.read_text();a="names=['modern_kimi_model_events_configure_r8669zq'";assert text.count(a)==1
text=text.replace(a,"names=['modern_kimi_loading_registry_work_r159','modern_kimi_loading_binding_counts_r9212','modern_kimi_W13_wavepartial_counts_r155','modern_kimi_model_events_configure_r8669zq'");p.write_bytes(text.encode())
p=M/'runtime/target_model.py';text=p.read_text();text=text.replace('from parameter_loader_r9143 import load_parameters','from parameter_loader_r9143 import load_parameters\nfrom model_registry_contract_r162 import validate_loaded_registry,W13OffProof')
text=text.replace("library,lib,providers=bind(manifest,b);report['actual_CAPI_providers']", "library,lib,providers=bind(manifest,b);w13_off_proof=W13OffProof(lib);report['actual_CAPI_providers']")
a="report.update(load_parameters(lib,ctx,model,text,keys,safe_open,torch,check,progress));report['head_after_finish']";assert text.count(a)==1
text=text.replace(a,"report.update(load_parameters(lib,ctx,model,text,keys,safe_open,torch,check,progress));report['loading_registry_qualification_r162']=validate_loaded_registry(report);report['head_after_finish']")
a="target_protocol.run(globals());assert report.get('serving_protocol_complete');";assert text.count(a)==1
text=text.replace(a,"target_protocol.run(globals());report['W13_default_OFF_r162']=w13_off_proof.finish();assert report.get('serving_protocol_complete');");p.write_bytes(text.encode())
# Keep protocol/math unmodified; six real-EOS pilot cases from original frozen questions.
queue=M/'queue'
for p in queue.glob('q*.json'):p.unlink()
fixture=V/'Kimi-W13-model-upgrade-private-r158/queue';chosen=[0,4,8,12,16,20];commands=[]
for i,n in enumerate(chosen):
 c=json.loads((fixture/f'q{n:04d}.json').read_bytes());c['id']=f'r162-pilot-{i}-C{c["C"]}';c['category']='registry-C1-pilot'if c['C']==1 else'registry-C8-pilot';c['model_profile']=False
 for key in list(c):
  if 'W13'in key or 'partial'in key:c.pop(key)
 commands.append(c);(queue/f'q{i:04d}.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
(queue/'q0006.json').write_text(json.dumps({'schema':1,'action':'stop','reason':'r162 six-case real-EOS pilot complete'})+'\n')
matrix={'case_count':6,'C1_cases':4,'C8_cases':2,'request_executions':20,'unique_question_ids':len({q['id']for c in commands for q in c['requests']}),'C1_is_subset_of_C8':True,'not_new32_heldout':True,'cap':512,'context':1024,'W':8,'strategy':'state','W2pair_fixed_ON':True,'W13_fixed_OFF':True,'registry_cache_fixed_ON':True,'model_profile_OFF':True,'EOS_not_forced':True,'root_only_GPU_GO':True}
(M/'fixed-data-r162.json').write_text(json.dumps(matrix,indent=2)+'\n')
p=M/'launcher/model_keeper.py';text=p.read_text().replace('frozen12 C8W8 GSM32 OFFONOFF, originalcleanup150, no newquality fromrepeats','frozen6 C1x4/C8x2 GSM16 realEOS pilot; cacheON W13OFF; originalcleanup150');p.write_bytes(text.encode())
# Retain all original lifecycle/network/owner/quality consistency gates; no sparse profile qualification is requested.
p=M/'launcher/model_audit.py';text=p.read_text();text=text.replace('case_summaries=[];RSS_by_node=[];','case_summaries=[];loading_attribution=[];RSS_by_node=[];')
a="profiles=d.get('sparse_model_profiles',[]);";assert text.count(a)==1
text=text.replace(a,"assert d['loading_registry_qualification_r162']['passed']and d['W13_default_OFF_r162']['passed'];loading_attribution.append({'node':node,**{k:d.get(k)for k in ['loading_seconds','loader_flushes','host_loader_phase_profile','registry_labels_r159','registry_total_8x8_r159','registry_finish_8x8_r159','finish_process_work_r159','H2D_host_observer_available_r159','H2D_DMA_time_r159','native_load_flush_profiles','loading_registry_qualification_r162','W13_default_OFF_r162']}});"+a)
begin=text.index('windows={');end=text.index('audit_result.update(',begin)
text=text[:begin]+"audit_result['profile_scope']='NOT_REQUESTED: loading diagnosis plus realEOS pilot';audit_result['loading_attribution_by_node']=loading_attribution\n"+text[end:]
text=text.replace('matrix12_complete','pilot6_complete').replace('33card cap8 micro1 currentC8W8 strict cacheOFF GSM32 OFFONOFF sparseprofile; no tokenidentitygate; EOS/caps/quality separate','33card fixed cacheON W13OFF; C1x4/C8x2 GSM16 reused screen; no tokenidentitygate; EOS/caps/quality separate')
a="out=r/'results'/";idx=text.index(a)
extra="users=[u for c in case_summaries for u in c['users']];audit_result['pilot_quality']={'request_executions':len(users),'correct':sum(u['quality']['correct']for u in users),'EOS':sum(u['EOS']for u in users),'truncated':sum(u['truncated']for u in users),'unique_question_ids':len({u['quality']['id']for u in users}),'not_new32_heldout':True};audit_result['Model_pilot_qualified']=audit_result['pilot6_complete']and len(users)==20 and all(u['quality']['correct']and u['EOS']and not u['truncated']for u in users)\n"
text=text[:idx]+extra+text[idx:];p.write_bytes(text.encode())
mp=M/'runtime/model-manifest-r9183.json';m=json.loads(mp.read_bytes());prefix='/refactor/incoming/Kimi-sparse-profile-Model-r162/'
m['native_DSO_SHA256']=receipt['native_SHA256'];m['loaded_library_SHA256'].pop('libKimi_W2_pair_r9280.so',None);m['loaded_library_SHA256'][Path(receipt['native']).name]=receipt['native_SHA256'];m['required_environment']['INFINILM_LOADING_BINDINGS_R9212']='1';m['required_environment']['INFINILM_KIMI_W13_WAVEPARTIAL_R155']='0'
for name in('loader_work_observer_r158.py','model_registry_contract_r162.py'):m['source_SHA256'][prefix+'runtime/'+name]=sha(M/'runtime'/name)
for path in m['source_SHA256']:
 if path.startswith(prefix):m['source_SHA256'][path]=sha(M/path[len(prefix):])
mp.write_text(json.dumps(m,indent=2)+'\n')
go=M/'launcher/GPU-GO-schema.json';g=json.loads(go.read_bytes());g['manifest_SHA256']=sha(mp);go.write_text(json.dumps(g,indent=2)+'\n')
pinpath=M/'launcher/model-source-pins.json';pins=json.loads(pinpath.read_bytes());pins['manifest_SHA256']=sha(mp);pins['native_DSO_SHA256']=receipt['native_SHA256']
pins['files']={prefix+str(f.relative_to(M)).replace('\\','/'):sha(f)for f in M.rglob('*')if f.is_file()and f!=pinpath}
pinpath.write_text(json.dumps(pins,indent=2)+'\n')
for p in M.rglob('*.py'):ast.parse(p.read_text())
out={'source_only':True,'GPU':False,'GPU_submitted':False,'CPU511_pass_not_Model_qualification':True,'CPU_build_SHA256':item['SHA256'],'native_SHA256':receipt['native_SHA256'],'manifest_SHA256':sha(mp),'files_SHA256':{str(f.relative_to(M)).replace('\\','/'):sha(f)for f in M.rglob('*')if f.is_file()},'parent_445_model_manifest_SHA256':'99d845cc08d0b13569e7ccd0ea253c27945391709f9175ceae4e845c44b15a14','root_publication_GO_required':True,'matrix':matrix}
(M/'producer-source-r162.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'prepared':True,'Model_files':len(out['files_SHA256']),'manifest_SHA256':sha(mp),'producer_SHA256':sha(M/'producer-source-r162.json'),'GPU':False}))
