"""Stdlib-only audit of immutable CPU979/981 source, MD, ELF logs and receipts.

Does not import candidate code, execute an ELF or contact the cluster.
Binary artifacts are identified by seal-time hashes; raw text is rehashed here.
"""
from pathlib import Path
import base64,gzip,hashlib,json,shlex,sys
H=lambda b:hashlib.sha256(b).hexdigest()
ARCHIVE='efabab5d822e03c1b340b55aa6c1411ea598f05620b48a8b6649729ebc0d5b40'
SOURCE='ca2a05cc0e20fa7670c1609183953466d500fd74c42554d72644ef43afd2fcaf'
PRODUCTION='7d992fbbcf05139bd15d82f75a9803009861e215e921e3057fa0ce9e28ce9161'
NATIVE='fa1f052fa08884a397cf2236daad775f8e1390babf7127a5941edae5c96571bb'
O='/refactor/incoming/Kimi-vocab-parallel-embedding-r461/'
D=O+'cpu-repair-v2/';B='/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2/'
OLD='/refactor/candidates/Kimi-vocab-parallel-embedding-r461-CPU-v1/'
def strong(s):
 return {w[-1]for line in s.splitlines()if len(w:=line.split())>=3 and w[-2]in ['T','D','B','R']}
def audit(path):
 t=json.loads(Path(path).read_bytes());z=base64.b64decode(t['gzip_base64']);assert H(z)==t['archive_SHA256']==ARCHIVE
 raw=gzip.decompress(z);assert H(raw)==t['uncompressed_SHA256']and len(raw)==t['uncompressed_bytes'];a=json.loads(raw)
 assert a['CPU']=='910407.981'and a['GPU']==a['Model']==0 and not a['Native_execution']and not a['SDK_execution']
 assert a['source_manifest_SHA256']==SOURCE and a['production_manifest_SHA256']==PRODUCTION and a['old979FAIL_preserved']
 fs=a['files'];assert len(fs)==t['rawfiles']==76 and len(a['artifacts'])==35
 for p,v in fs.items():b=v['rawtext'].encode();assert H(b)==v['SHA256']and len(b)==v['bytes'],p
 get=lambda p:fs[p]['rawtext'];js=lambda p:json.loads(get(p))
 for root,h in [(O,PRODUCTION),(D,SOURCE)]:
  assert fs[root+'CPU_SOURCE_MANIFEST.json']['SHA256']==h
  for p,h in js(root+'CPU_SOURCE_MANIFEST.json')['files_SHA256'].items():assert fs[root+p]['SHA256']==h,p
 q=js(B+'CPU-result.json');old=js(OLD+'CPU-result.json')
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.981'and q['source_manifest_SHA256']==SOURCE
 assert q['GPU']==q['Model']==0 and not q['Native_dlopen']and not q['SDK_execution']and not q['checkpoint_weight_payload']and not q['Torch_imported']
 assert not old['passed']and old['actual_CPU']=='910407.979'and 'vocab_embedding_r461' in get(OLD+'pair_probe-compile.log')
 assert not old.get('compiled')and old['actual_source_CPU_contract']['passed']and old['actual_Python_loader_contract']['passed']
 reused=q['reused_CPU979_contracts'];assert reused['executed_again']is False
 for k in ['actual_source_CPU_contract','actual_Python_loader_contract','recording_source_identity']:assert reused[k]==old[k]
 assert fs[OLD+'CPU-result.json']['SHA256']==reused['receipt']['SHA256']
 for label in ['actual-Weight-Embedding-source-recording-contract','actual-loader-main-recording-contract']:
  assert old['logs'][label]['rc']==0 and fs[OLD+label+'.log']['SHA256']==old['logs'][label]['SHA256']
 assert old['actual_source_CPU_contract']['all32_loader_slices']==33 and old['actual_source_CPU_contract']['boundary_checks']==455 and old['actual_source_CPU_contract']['rejections']==6
 assert old['actual_Python_loader_contract']['registered_ranks']==32 and old['actual_Python_loader_contract']['actual_flushes']==2
 assert get(D+'native/pair_probe.cpp')==get(O+'native/pair_probe.cpp').replace('struct Context{','struct Context{bool vocab_embedding_r461=false;',1)
 assert get(D+'native/pair_probe.cpp').count('struct Context{bool vocab_embedding_r461=false;')==1
 assert q['source_contract']['actual_private_Context_flag_declared']and q['source_contract']['new_DSO_required']and not q['source_contract']['old_DSO_qualification_inherited']
 prod=[c for c in q['commands']if '-c'in c];assert len(prod)==3 and [Path(c[c.index('-c')+1]).name for c in prod]==['pair_probe.cpp','kimi_k3_text_model.cpp','masked_embedding.cu']
 assert q['new_HOST_TUs']==2 and q['new_GPU_TUs']==1 and q['pure_source_contract_TUs']==0
 for label,v in q['logs'].items():assert v['rc']==0 and fs[B+label+'.log']['SHA256']==v['SHA256'],label
 for unit,v in q['compiled'].items():
  dep=lambda p:set(shlex.split(get(p).replace('\\\n',' ').split(':',1)[1]))
  assert dep(B+unit+'.d')==dep(B+unit+'.before.d')==set(v['all_new_MD'])and v['MD_before_after_identical']
  assert fs[v['source']]['SHA256']==v['source_SHA256']and a['artifacts'][B+unit+'.o']['SHA256']==v['object_SHA256']
  for p,h in v['all_new_MD'].items():
   if p in fs:assert fs[p]['SHA256']==h,p
 assert O+'vocab_parallel_embedding_r461.hpp'in q['compiled']['text_model']['all_new_MD']
 assert O+'embedding_index_r461.hpp'in q['compiled']['masked_embedding']['all_new_MD']
 assert D+'native/pair_probe.cpp'in q['compiled']['pair_probe']['all_new_MD']
 parent=js('/refactor/candidates/Kimi-AR-persistent-storage-optin-r456-CPU-v2/CPU-result.json')
 assert len(q['all_linked_object_SHA256'])==34 and len(q['unchanged_objects_SHA256'])==31
 for p,h in q['all_linked_object_SHA256'].items():assert a['artifacts'][p]['SHA256']==h
 for p,h in q['unchanged_objects_SHA256'].items():assert parent['all_linked_object_SHA256'][p]==h
 assert q['actual_dependency_library_SHA256']==parent['actual_dependency_library_SHA256']
 for p,h in q['old_class_header_ABI_source_pins'].items():assert fs[p]['SHA256']==h
 s0=strong(get(B+'parent-strong-symbols.log'));s1=strong(get(B+'new-strong-symbols.log'))
 assert s0<=s1 and len(s0)==q['old_strong_exports_retained']==392
 assert len({s for s in s0 if s.startswith('modern_')})==q['old_CAPI_retained']==136
 assert set(q['new_CAPI'])<=s1 and len(q['new_CAPI'])==4
 assert 'SYMBOLIC'in get(B+'Native-dynamic.log')
 ldd=get(B+'Native-ldd-relocations-only.log');assert 'undefined symbol'not in ldd and 'not found'not in ldd
 assert a['artifacts'][q['native']]['SHA256']==q['native_SHA256']==NATIVE
 assert '910407.981|Kimi461-vocab-parallel-embedding-CPU-v2|COMPLETED|0:0|00:00:47|a14r2n20'in get('accounting979981')
 assert '910407.979|Kimi461-vocab-parallel-embedding-CPU-v1|FAILED|1:0'in get('accounting979981')
 assert not a['owned_srun_PID408969_cmdline']or 'Kimi461-vocab-parallel-embedding-CPU-v2'not in a['owned_srun_PID408969_cmdline']
 return dict(schema=1,passed=True,actual_CPU='910407.981',terminal='COMPLETED0/47s',source_manifest_SHA256=SOURCE,production_manifest_SHA256=PRODUCTION,
  sealed_archive_SHA256=ARCHIVE,rawfiles_verified=len(fs),binary_artifact_hash_records=35,native=q['native'],native_SHA256=NATIVE,
  compiled_HOST_TUs=2,compiled_GPU_TUs=1,reused_objects=31,old_strong_exports_retained=392,old_CAPI_retained=136,new_CAPI=q['new_CAPI'],
  CPU979_FAIL_preserved=True,CPU979_two_passed_recording_consumers_reused_not_executed_again=True,actual_source_Weight_checks=33,address_boundary_checks=455,rejection_checks=6,
  private_Context_one_field_delta_verified=True,shared_class_headers_unchanged=True,Native_dlopen=False,SDK_execution=False,GPU=0,Model=0,GPU_GO=False,Model_GO=False,
  limits=['Weight/LoadBatch and Python loader source consumers use recording Tensor/Runtime/safe_open leaves; no actual checkpoint payload.',
          'MD/ELF/symbol/dependency closure is compile/link evidence, not device execution.',
          'Source saving2275409920B/rank does not establish physical floor, collective numeric behavior, actual allocator residency or Model performance.'])
if __name__=='__main__':
 out=audit(sys.argv[1]);s=json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
 if len(sys.argv)>2:Path(sys.argv[2]).write_text(s,encoding='utf-8',newline='\n')
 print(s,end='')
