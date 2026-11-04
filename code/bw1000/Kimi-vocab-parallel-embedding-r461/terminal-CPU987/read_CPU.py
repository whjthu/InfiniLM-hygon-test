"""Offline stdlib audit: CPU985 failure +986 timeout +987 conditional link.

No candidate imports, compiler, Native/SDK execution or cluster access.
"""
from pathlib import Path
import base64,gzip,hashlib,json,shlex,sys
H=lambda b:hashlib.sha256(b).hexdigest()
ARCHIVE='e878c18ec3c5864559a02c9e785ab3eeb7c5fcf3ab69f442fb34115500895f39'
SOURCE='333eab8494b17ed93db3d14bd197a12ac2711094ac427d1573002709b2191506'
PROBE='108007a29dff5737c2746f8a97e126a7ab16d663f57828557ceb1e0757760231'
NATIVE='bbc2dcda0256446091ea09ee6d9cf0efde9d9e44b02e1e999d87ac58abb671f9'
P='/refactor/incoming/Kimi-vocab-parallel-embedding-r461/native-probe-v1/'
D=P+'link-repair-v2/';B='/refactor/candidates/Kimi-vocab-parallel-embedding-r461-probe-link-CPU-v2/'
O='/refactor/candidates/Kimi-vocab-parallel-embedding-r461-probe-CPU-v1/'
def symbols(s):return{w[-1]for line in s.splitlines()if len(w:=line.split())>=3 and w[-2]in ['T','D','B','R']}
def audit(path):
 t=json.loads(Path(path).read_bytes());packed=base64.b64decode(t['gzip_base64']);assert H(packed)==t['archive_SHA256']==ARCHIVE
 raw=gzip.decompress(packed);assert len(raw)==t['uncompressed_bytes']and H(raw)==t['uncompressed_SHA256'];a=json.loads(raw)
 assert a['CPU']=='910407.987'and a['GPU']==a['Model']==0 and not a['Native_execution']and not a['SDK_execution']and a['old985FAIL_and_986TIMEOUT_preserved']
 fs=a['files'];assert len(fs)==t['rawfiles']==58 and len(a['artifacts'])==36
 for p,v in fs.items():b=v['rawtext'].encode();assert len(b)==v['bytes']and H(b)==v['SHA256'],p
 get=lambda p:fs[p]['rawtext'];js=lambda p:json.loads(get(p))
 for root,h in [(P,PROBE),(D,SOURCE)]:
  assert fs[root+'CPU_SOURCE_MANIFEST.json']['SHA256']==h
  for p,h in js(root+'CPU_SOURCE_MANIFEST.json')['files_SHA256'].items():assert fs[root+p]['SHA256']==h,p
 q=js(B+'CPU-result.json');old=js(O+'CPU-result.json');pins=js(D+'PARENT.json')
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.987'and q['source_manifest_SHA256']==SOURCE
 assert not q['compile']and q['new_HOST_TUs']==q['new_GPU_TUs']==q['GPU']==q['Model']==0 and not q['Native_dlopen']and not q['SDK_execution']
 assert q['metadata_completed']and q['link_started']and q['original985FAIL_preserved']and q['metadata986TIMEOUT_preserved']
 assert not old['passed']and old['actual_CPU']=='910407.985'and old['phase']=='worker-compile'
 assert 'old MD changed/omitted'in old['error']and old['source_contract']['passed']
 assert fs[O+'CPU-result.json']['SHA256']==pins['CPU985_receipt']['SHA256']
 for label in ['pair_probe-compile','worker-compile','actual-fixed-sidecar-contract']:
  assert old['logs'][label]['rc']==0 and fs[O+label+'.log']['SHA256']==old['logs'][label]['SHA256']
 contract=json.loads(get(O+'actual-fixed-sidecar-contract.log'))
 assert contract['passed']and contract['cases']==6 and contract['actual_std_threads']==8 and not contract['SDK_execution']and not contract['Runtime_or_Tensor_executed']
 assert q['successful_CPU985_contract_reused_not_rerun']and q['original985_objects_reused_not_recompiled']
 assert len([c for c in old['commands']if '-c'in c])==2
 assert len([c for c in old['commands']if any(p.endswith('.cpp')for p in c)and '-M'not in c])==3 # pure contract compile+link, two production HOST
 assert not any('-c'in c or any(v.endswith(('.cpp','.cu'))for v in c)for c in q['commands'])
 before=js(B+'metadata-before-link.json');assert before['metadata_completed']and not before['link_started']and not before['passed']
 assert before['explicit_alias_proof']==q['explicit_alias_proof']and before['original985_metadata']==q['original985_metadata']
 aliases=js(D+'EXACT_ALIASES.json');assert len(aliases)==len(q['explicit_alias_proof'])==26
 for row in aliases:
  proof=q['explicit_alias_proof'][row['old_path']]
  assert proof['actual_path']==row['actual_path']and proof['SHA256']==row['SHA256']and proof['samefile']
  for k in ['old_stat','actual_stat']:assert proof[k]['SHA256']==row['SHA256']
  assert (proof['old_stat']['device'],proof['old_stat']['inode'])==(proof['actual_stat']['device'],proof['actual_stat']['inode'])
  assert proof['canonical'].startswith('/refactor/')or proof['canonical'].startswith('/public/')
 base=js('/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2/CPU-result.json')
 parent=js('/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v2/CPU-result.json')
 counts={}
 for unit,receipt,oldmd in [('pair_probe',base,base['compiled']['pair_probe']['all_new_MD']),('worker',parent,parent['HOST']['worker']['all_new_MD'])]:
  v=q['original985_metadata'][unit]
  dep=lambda s:set(shlex.split(s.replace('\\\n',' ').split(':',1)[1]))
  actual=dep(get(O+unit+'.d'));assert actual==dep(get(O+unit+'.before.d'))==set(v['actual_MD_paths_SHA256'])and v['MD_before_after_same']
  for suffix in ['.d','.before.d']:
   assert fs[O+unit+suffix]['SHA256']==pins['original_MD'][unit+suffix]['SHA256']==v['original_MD'][suffix]['SHA256']
   assert v['original_MD'][suffix]['rawtext']==get(O+unit+suffix)
  origin=next(c[c.index('-c')+1]for c in receipt['commands']if '-c'in c and Path(c[c.index('-c')+1]).name==unit+'.cpp')
  mapped=set()
  for p,h in oldmd.items():
   if p==origin:continue
   target=p if p in actual else q['explicit_alias_proof'][p]['actual_path']
   assert v['actual_MD_paths_SHA256'][target]==h;mapped.add(target)
  assert actual-mapped==set(pins['exact_additional_MD'][unit])
  for p,h in pins['exact_additional_MD'][unit].items():assert v['actual_MD_paths_SHA256'][p]==h
  assert fs[v['source']['path']]['SHA256']==v['source']['SHA256']==H(v['source']['rawtext'].encode())
  argv=v['actual_compile_argv'];assert argv in old['commands']and argv[argv.index('-c')+1]==v['source']['path']
  assert argv[argv.index('-o')+1]==v['object']['path']and a['artifacts'][v['object']['path']]=={k:v['object'][k]for k in ['bytes','SHA256']}
  assert v['object']==pins['objects'][unit];counts[unit]=len(actual)
 assert counts==dict(pair_probe=566,worker=575)
 for p,h in q['all_linked_object_SHA256'].items():assert a['artifacts'][p]['SHA256']==h
 assert len(q['all_linked_object_SHA256'])==34
 link=next(c for c in q['commands']if '-shared'in c);assert '-c'not in link
 assert sorted(v for v in link if v.endswith('.o'))==sorted(q['all_linked_object_SHA256'])
 replacements={'/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2/pair_probe.o':O+'pair_probe.o','/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v2/worker.o':O+'worker.o'}
 original_link=next(c[:]for c in base['commands']if '-shared'in c)
 expected=[replacements.get(v,v)for v in original_link];expected[expected.index('-o')+1]=q['native'];assert link==expected
 for p,h in base['all_linked_object_SHA256'].items():
  if p not in replacements:assert q['all_linked_object_SHA256'][p]==h
 for label,v in q['logs'].items():assert v['rc']==0 and fs[B+label+'.log']['SHA256']==v['SHA256']
 oldsym=symbols(get(B+'CPU981-strong-symbols.log'));newsym=symbols(get(B+'new-strong-symbols.log'))
 assert oldsym<=newsym and len(oldsym)==q['old_strong_exports_retained']==396
 assert len({s for s in oldsym if s.startswith('modern_')})==q['old_CAPI_retained']==140
 assert set(q['new_CAPI'])=={'modern_kimi_embedding_probe_load_r461','modern_kimi_embedding_probe_r461'}<=newsym
 assert 'SYMBOLIC'in get(B+'Native-dynamic.log')
 assert 'undefined symbol'not in get(B+'Native-ldd-relocations-only.log')and 'not found'not in get(B+'Native-ldd-relocations-only.log')
 assert q['native_SHA256']==a['artifacts'][q['native']]['SHA256']==NATIVE
 progress=[json.loads(s)for s in get(B+'progress.jsonl').splitlines()]
 phases=[p['phase']for p in progress];assert phases.index('all-identity-PASS-before-link')<phases.index('private-Native-link')
 assert all(left['elapsed_s']<=right['elapsed_s']for left,right in zip(progress,progress[1:]))
 accounting=get('accounting985986987')
 for s in ['910407.985|Kimi461-vocab-parallel-embedding-probe-CPU-v1|FAILED|1:0|00:00:40','910407.986|Kimi461-CPU985-MD-samefile-only|TIMEOUT|0:15|00:01:00','910407.987|Kimi461-probe-link-only-CPU-v2|COMPLETED|0:0|00:00:03']:assert s in accounting
 return dict(schema=1,passed=True,actual_CPU='910407.987',terminal='COMPLETED0/3s',source_manifest_SHA256=SOURCE,probe_source_manifest_SHA256=PROBE,
  archive_SHA256=ARCHIVE,rawfiles_verified=58,artifact_hash_records=36,compile=False,successful_CPU985_HOST_objects_reused=2,CPU981_objects_reused=32,
  actual_container_samefile_aliases_verified=26,full_MD_paths=counts,metadata_identity_completed_before_link=True,
  CPU985_FAIL_preserved=True,metadata_CPU986_TIMEOUT_preserved=True,CPU985_successful_sidecar_six_cases_reused_not_rerun=True,
  native=q['native'],native_SHA256=NATIVE,old_strong_exports_retained=396,old_CAPI_retained=140,new_CAPI=q['new_CAPI'],
  new_private_Context_requires_new_DSO=True,Native_dlopen=False,SDK_execution=False,GPU=0,Model=0,GPU_GO=False,Model_GO=False,
  limits=['CPU987 proves engineering metadata/ABI/link closure. Actual masked lookup, original TP32 BF16 collective numerical behavior and physical floor remain unqualified.',
   'CPU985 compiler binary was not hashed contemporaneously. Actual compile argv/source/object identity and current same-image compiler hash are separately recorded.',
   'CPU986 timeout yielded no JSON. CPU987 successful focused metadata does not retroactively prove its timeout cause.',
   'No Model weights or full forward/HTTP/quality/throughput were executed. Source memory saving is not observed driver free.'])
if __name__=='__main__':
 result=audit(sys.argv[1]);s=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if len(sys.argv)>2:Path(sys.argv[2]).write_text(s,encoding='utf-8',newline='\n')
 print(s,end='')
