"""Read-only qualification of the actual885 linked artifact; no compile or link."""
from pathlib import Path,PurePosixPath
import hashlib,json,os,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-qualified-v3')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()

def exact_dependency(text,expected):
 assert PurePosixPath(expected).is_absolute()
 assert 'not found'not in text and 'undefined symbol'not in text
 rows=[]
 for line in text.splitlines():
  fields=line.split()
  if len(fields)==2 and fields[0]==expected and fields[1].startswith('(0x')and fields[1].endswith(')'):rows.append(dict(kind='exact_absolute_DT_NEEDED',path=fields[0],raw_line=line))
  elif len(fields)==4 and fields[0]==PurePosixPath(expected).name and fields[1]=='=>'and fields[2]==expected and fields[3].startswith('(0x')and fields[3].endswith(')'):rows.append(dict(kind='exact_soname_arrow',path=fields[2],raw_line=line))
 assert len(rows)==1,'Missing, duplicate, unexpected path or malformed helper ldd line'
 return rows[0]

def main():
 assert not B.exists(),'Read original qualifier only';B.mkdir()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Native_dlopen=False,weights=0,compile=0,link=0,objcopy=0,Runtime_created=False,CPU885_original_FAIL_preserved=True)
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R437_QUALIFIER_SHA'];m=json.loads(mf.read_bytes());q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(mf))
  for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
  c=json.loads((D/'REUSE_CPU885.json').read_bytes())
  for root in c['source_roots']:
   directory=Path(root['path']);assert sha(directory/'SOURCE_MANIFEST.json')==root['SHA256']
   for p,h in json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(directory/p)==h,p
  receipt=Path(c['receipt']['path']);assert sha(receipt)==c['receipt']['SHA256'];old=json.loads(receipt.read_bytes())
  assert old['actual_CPU']=='910407.885'and not old['completed']and not old['passed']and old['compile']==0 and old['link']==1
  assert old['error']=='AssertionError()'and 'rows=[line.split()for line in ldd.splitlines()'in old['traceback']
  commands=old['commands'];assert len(commands)==12 and all(v['rc']==0 for v in commands)
  for v in commands:assert sha(v['log'])==v['SHA256'],v['label']
  artifact=c['library'];assert sha(artifact['path'])==artifact['SHA256']
  localized=old['localized_private_object'];assert localized['text_bytes_unchanged']and localized['all_other_global_symbols_unchanged']and len(localized['only_localized'])==7 and len(localized['all_text_sections_SHA256'])==564
  assert sha(localized['object'])==localized['object_SHA256']and sha(localized['original'])==localized['original_SHA256']
  for label in ['old-Native-exports','original-strong-MoE','dynamic-defined','actual-dynamic','actual-ldd-relocations']:assert any(v['label']==label for v in commands)
  ldd=Path(next(v['log']for v in commands if v['label']=='actual-ldd-relocations')).read_text()
  dynamic=Path(next(v['log']for v in commands if v['label']=='actual-dynamic')).read_text();assert 'SYMBOLIC'in dynamic and '['+c['helper']['path']+']'in dynamic
  match=exact_dependency(ldd,c['helper']['path']);assert sha(match['path'])==c['helper']['SHA256']
  # Same final parser on both real syntax forms and explicit path/duplicate negatives.
  assert exact_dependency(c['helper']['path']+' (0x1)',c['helper']['path'])['path']==c['helper']['path']
  assert exact_dependency(PurePosixPath(c['helper']['path']).name+' => '+c['helper']['path']+' (0x1)',c['helper']['path'])['path']==c['helper']['path']
  rejected=0
  for bad in ['/wrong/'+PurePosixPath(c['helper']['path']).name+' (0x1)',c['helper']['path']+' (0x1)\n'+c['helper']['path']+' (0x2)','libkimi_grouped_prefill.so => not found']:
   try:exact_dependency(bad,c['helper']['path'])
   except AssertionError:rejected+=1
  assert rejected==3
  parent=Path(c['source_roots'][0]['path']);contract=json.loads((parent/'BUILD_CONTRACT.json').read_bytes());native=contract['parent_native']
  for name,expected in native['actual_dependency_library_SHA256'].items():
   rows=[line.split()for line in ldd.splitlines()if line.split()and line.split()[0]==name];assert len(rows)==1 and rows[0][1]=='=>'and sha(rows[0][2])==expected['SHA256']
  symbols=lambda s:{v.split()[-1]:v.split()[-2]for v in s.splitlines()if len(v.split())>=3}
  old_exports=symbols(Path(next(v['log']for v in commands if v['label']=='old-Native-exports')).read_text());exports=symbols(Path(next(v['log']for v in commands if v['label']=='dynamic-defined')).read_text());required={k:v for k,v in old_exports.items()if k.startswith('modern_')and v in ['T','D','B','R']};assert required and all(exports.get(k)==v for k,v in required.items())
  for name in ['modern_kimi_lowrank_full_moe_r437','modern_kimi_lowrank_full_moe_result_r437','kimi_lowrank_gemv_r432']:assert exports.get(name)=='T'
  q.update(completed=True,passed=True,library=artifact['path'],library_SHA256=artifact['SHA256'],actual_original_885_libraries_and_checks_reused=True,helper_binding=match,helper_SHA256=c['helper']['SHA256'],old_strong_CABI_preserved=len(required),all564_text_segments_and_other_globals_unchanged=True,localized_private_object=localized,source_math_or_Class_header_changed=False,old883_and885_receipts=c['old_failures'],qualifier_scope='Only strict actual absolute DT_NEEDED line parsing; original885 compile/link/nm/ldd and883 source/FFI are retained evidence, not re-executed or reclassified as successful jobs',actual_operator_or_TP16_numeric=False,Model_GO=False,GPU_GO=False)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
