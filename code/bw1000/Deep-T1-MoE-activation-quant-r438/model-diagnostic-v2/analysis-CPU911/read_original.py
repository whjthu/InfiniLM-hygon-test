"""Stdlib audit of original CPU911; no Native/SDK dlopen or original suite replay."""
import argparse,hashlib,json,re,shlex
from pathlib import Path
H=lambda raw:hashlib.sha256(raw).hexdigest()
def analyze(archive,source):
 files=archive['files'];by={}
 for v in files:
  raw=v['text'].encode();assert len(raw)==v['bytes']and H(raw)==v['SHA256'];assert v['path']not in by;by[v['path']]=v
 def item(suffix):
  rows=[v for p,v in by.items()if p.endswith(suffix)];assert len(rows)==1,(suffix,len(rows));return rows[0]
 receipt=item('/CPU-build.json');q=json.loads(receipt['text']);assert receipt['SHA256']=='cbf33261cf3d64aa496a4b8ad0fd2a3c2fcefa62ad36465e9168a0475920896d'
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.911'and q['source_manifest_SHA256']=='cf570e363016fac276037184ffdc3bdcd03f3bc3b8896ddae1c1031e0d3b552d'
 assert not any(q[k]for k in ['Native_dlopen','Runtime_created','weights_payload_opened','GPU_initialized','SDK_initialized','Model_loaded','Model_Runtime_constructor_executed','actual_Model_forward_executed'])
 m=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert H((source/'SOURCE_MANIFEST.json').read_bytes())==q['source_manifest_SHA256']
 assert all(H((source/n).read_bytes())==h for n,h in m['files_SHA256'].items())and not m['GPU_GO']and not m['Model_GO']
 assert q['production_HOST_compiles']==2 and q['pure_scope_HOST_compiles']==0 and q['GPU_compiles']==0 and not q['old_CPU902_904_suites_replayed']
 commands=q['commands'];compiles=[c for c in commands if '-c'in c];assert len(compiles)==2 and all(Path(c[0]).name=='g++'for c in compiles)and all(not any('.cu'==Path(x).suffix for x in c)for c in compiles)
 assert {Path(c[c.index('-c')+1]).name for c in compiles}=={'model_moe.cpp','model_bank.cpp'}
 assert len(q['compiled'])==2 and q['Native_object_count']==49 and len(q['all_objects_SHA256'])==49 and q['unchanged_actual_Model_objects']==46
 for u in q['compiled']:
  md=item('/'+Path(u['MD']).name);assert md['SHA256']==u['MD_SHA256'];deps=set(shlex.split(md['text'].replace('\\\n',' ').split(':',1)[1]));assert deps==set(u['dependencies_SHA256'])and u['fresh_M_equals_MD']
  own={Path(p).name:h for p,h in u['dependencies_SHA256'].items()if '/model-diagnostic-v2/'in p}
  assert own.get('capture.hpp')==H((source/'capture.hpp').read_bytes())
  assert own.get('model_scope.hpp')==H((source/'model_scope.hpp').read_bytes())
 assert q['reused_HIP']['SHA256']=='7a031d5a2627650ced94da012abc7a36b867770a63d6628952d2d496b8d47c5a'
 def syms(suffix):return {v.split()[-1]:v.split()[-2]for v in item(suffix)['text'].splitlines()if len(v.split())>=3}
 old,new=syms('/parent-Model902-defined.log'),syms('/new-Model-defined.log');weak={'W','w','V','v','u'};strong={n for n,k in old.items()if k not in weak};cexports={n for n in old if not n.startswith('_Z')}
 assert strong<=set(new)and cexports<=set(new)and not set(old)-set(new)
 required={'deep_model_capture_'+n+'_r438v2'for n in ['arm','seal','clear','abandon','count','record','tensor','copy']}|{'deep_model_actquant_'+n+'_r438'for n in ['set','clear','read','metadata']};assert required<=set(new)
 ff={r.split()[-1]for r in item('/actual-unchanged-FFN-required-types.log')['text'].splitlines()if 'DeepseekV4MoERoutedR20'in r};assert len(ff)>=3 and ff<=set(new)
 needed=lambda suffix:re.findall(r'\(NEEDED\).*?\[(.*?)\]',item(suffix)['text']);assert sorted(needed('/parent-Model902-dynamic.log'))==sorted(needed('/new-Model-dynamic.log'))
 ld=item('/new-Model-ldd-r-NO-initializers.log')['text'];assert 'not found'not in ld and 'undefined symbol'not in ld
 consumer=json.loads(item('/actual-new-typed-save-teacher-consumer.log')['text']);assert consumer==q['actual_changed_save_consumer']and consumer['completed']and consumer['passed']and not consumer['actual_Native_execution']and not consumer['actual_Model_activation_captured']and consumer['GPU']==0
 cases={v['name']:v for v in consumer['cases']};assert set(cases)=={'normal','forward_nonzero','shadow_diff','bad_dtype'}
 assert cases['normal']['actual_forward_attempts']==cases['normal']['completed_forwards']==368 and cases['normal']['resets']==4 and cases['normal']['raw_budget_used']==114496768<128*1024**2
 assert cases['forward_nonzero']['actual_forward_attempts']==12 and cases['forward_nonzero']['completed_forwards']==11 and cases['forward_nonzero']['resets']==2
 assert cases['shadow_diff']['completed_forwards']==91 and cases['bad_dtype']['completed_forwards']==91 and cases['bad_dtype']['raw_budget_used']>0
 assert any(r[0]=='910407.911'and r[2:4]==['COMPLETED','0:0']for r in archive['accounting'])
 return dict(passed=True,scope='Original CPU911 source/link/typed-save recordings only',actual_CPU='910407.911',receipt_SHA256=receipt['SHA256'],receipt_bytes=receipt['bytes'],source_manifest_SHA256=q['source_manifest_SHA256'],library=q['library'],library_SHA256=q['library_SHA256'],Native_objects=49,original_Model_objects_retained=46,reused_HIP_unchanged=True,changed_HOST_compiles=2,new_GPU_compiles=0,old_strong_exports_retained=len(strong),old_C_exports_retained=len(cexports),new_capture_CABI=sorted(required),nonpublic_weak_removed=[],DT_NEEDED_unchanged=True,actual_FFN_original_namespace_symbols_resolved=True,changed_save_cases=consumer['cases'],full_teacher_plan_verified_in_recording=True,recording_owner_labels_not_Runtime_threads=True,actual_Native_or_GPU_or_Model_or_activation_executed=False,GPU_GO=False,Model_GO=False,root_cause='UNKNOWN',future_entry_and_device_qualification_required=True,verified_original_files=len(files))
def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args();raw=a.archive.read_bytes();q=json.loads(raw)
 if 'sealed_archive'in q:
  seal=q['sealed_archive'];raw=seal['text'].encode();assert len(raw)==seal['bytes']and H(raw)==seal['SHA256'];q=json.loads(raw)
 out=analyze(q,a.source);out['archive_SHA256']=H(raw);text=json.dumps(out,indent=2)+'\n'
 if a.output:a.output.write_text(text,encoding='utf-8')
 print(text,end='')
if __name__=='__main__':main()
