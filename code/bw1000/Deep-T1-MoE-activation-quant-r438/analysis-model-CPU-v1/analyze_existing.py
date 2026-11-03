"""Stdlib independent CPU902 log/object-graph closure; never loads a library."""
from pathlib import Path
import argparse,hashlib,json
SHA=lambda b:hashlib.sha256(b).hexdigest()
ARCHIVE='2476736da12d2bee0f32f12d7fce49c1063ede1a75c714e72f3341bdffe7fe98'
RECEIPT='ea076c6762f2e1ac1fad3db65608170bf4ae0eb4af860fd3765abc87e2e55bcd'
BASE_RECEIPT='d9df817a0c9f9b770dff497a5cf754536004bb3308a62432e0464ca84f935055'
def symbols(text):return {line.split()[-1]:line.split()[-2]for line in text.splitlines()if len(line.split())>=3}
def needed(text):return[line.split('[',1)[1].split(']',1)[0]for line in text.splitlines()if'NEEDED'in line]
def analyze(archive,baseline):
 raw=Path(archive).read_bytes();wrapped=json.loads(raw)
 if'sealed_archive'in wrapped:raw=wrapped['sealed_archive']['text'].encode();assert SHA(raw)==wrapped['sealed_archive']['SHA256']and len(raw)==wrapped['sealed_archive']['bytes']
 assert SHA(raw)==ARCHIVE;a=json.loads(raw);assert a['schema']=='Deep438-Model-CPU902-artifact-closure-v1'and a['original_CPU']=='910407.902';files={}
 for f in a['files']:
  b=f['text'].encode();assert len(b)==f['bytes']and SHA(b)==f['SHA256'];name=Path(f['path']).name;assert name not in files;files[name]=b
 assert len(files)==11 and SHA(files['CPU-build.json'])==RECEIPT;q=json.loads(files['CPU-build.json']);assert q['completed']and q['passed']and q['actual_CPU']=='910407.902'and q['source_manifest_SHA256']=='34dd10c7e75e1a525332ffaf39609387f08c350f1ef3e046d4b6898b53cbfd27'
 assert all(q[k]is False for k in ['Native_dlopen','Runtime_created','weights_payload_opened','GPU_initialized','SDK_initialized','Model_loaded','GPU_GO','Model_GO'])
 b=json.loads(Path(baseline).read_bytes());parent=next(f for f in b['files']if f['path'].endswith('CPU-build.json'));assert SHA(parent['text'].encode())==parent['SHA256']==BASE_RECEIPT;p=json.loads(parent['text']);assert p['passed']and p['actual_CPU']=='910407.889'
 original=dict(p['Native_unchanged_objects_SHA256']);original[p['Native_worker']['new_object']]=p['Native_worker']['new_object_SHA256'];assert len(original)==48
 removed=['/refactor/candidates/deep-owner-device-r8918/deep_model_owner_r8900.cpp.o','/refactor/candidates/deep-quant1-r8881/deepseek_v4_int8_mlp_bank_r20.cpp.o'];assert original[removed[0]]=='43f97ef93315f43e5332d85d7807022d2d6065e8a46e0b33c40498d48de8ddb0'and original[removed[1]]=='b51baf55c43708044aa8f076414b02d782e3ee282f190df24aad44bba1ee1152'
 retained={k:v for k,v in original.items()if k not in removed};assert len(retained)==46
 compiled=q['compiled'];assert len(compiled)==3 and all(v['fresh_M_equals_MD']for v in compiled)
 prod={r['object']:r['object_SHA256']for r in compiled if Path(r['object']).name in['model_moe.o','model_bank.o']};assert len(prod)==2
 scope=[r for r in compiled if Path(r['object']).name=='scope.o'];assert len(scope)==1
 hip=q['reused_HIP'];assert hip['object']==next(r['object']for r in p['compiled']if r['object'].endswith('activation_quant.o'))
 assert hip['SHA256']==next(r['object_SHA256']for r in p['compiled']if r['object']==hip['object'])
 expected={**retained,**prod,hip['object']:hip['SHA256']};assert q['all_objects_SHA256']==expected and len(expected)==49 and q['Native_object_count']==49 and q['unchanged_actual_Model_objects']==46
 compile_commands=[c for c in q['commands']if'-c'in c];assert len(compile_commands)==3 and all(c[0]=='g++'for c in compile_commands)and q['GPU_compiles']==0
 old=symbols(files['old-afc6-defined.log'].decode());new=symbols(files['new-Model-defined.log'].decode());weak={'W','w','V','v','u'};strong={n for n,k in old.items()if k not in weak};Cexports={n for n in old if not n.startswith('_Z')};assert strong<=set(new)and Cexports<=set(new)
 required=['deep_model_actquant_'+n+'_r438'for n in ['set','clear','read','metadata']];assert all(new.get(n)=='T'for n in required)
 missing=sorted(set(old)-set(new));assert missing==q['nonpublic_weak_removed_diagnostic']and all(old[n]in weak and n.startswith('_Z')for n in missing)
 required_types={line.split()[-1]for line in files['actual-unchanged-FFN-required-types.log'].decode().splitlines()if'DeepseekV4MoERoutedR20'in line};assert len(required_types)==3 and required_types<=set(new)and sorted(required_types)==q['actual_unchanged_FFN_original_MoE_symbols_resolved']
 assert all('actquant_r438'not in name for name in required_types)
 before=needed(files['old-afc6-dynamic.log'].decode());after=needed(files['new-Model-dynamic.log'].decode());assert sorted(before)==sorted(after)and after==q['DT_NEEDED'];assert not any('libDeep_'in name for name in after)
 ld=files['new-Model-ldd-r-NO-initializers.log'].decode();assert'not found'not in ld and'undefined symbol'not in ld
 pure=needed(files['scope-dynamic.log'].decode());assert set(pure)<={'libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6','libpthread.so.0','ld-linux-x86-64.so.2'}
 pure_ld=files['scope-ldd.log'].decode();assert not any(s in pure_ld.lower()for s in ['libinfini','libdeep','libtorch','libcuda','libhip','libgalaxy','flash_attn'])
 contract=json.loads(files['actual-pure-Model-scope-8thread.log'].decode().splitlines()[-1]);assert contract==q['actual_scope_CPU_contract']and contract['passed']and contract['owner_threads']==8 and contract['default_unarmed_OFF']and contract['T2_fallback']and contract['original_failure_unwinds']and contract['whole_engine_quiescence_is_caller_obligation']
 consumer=json.loads(files['actual-quiescent-request-consumer.log'].decode().splitlines()[-1]);assert consumer==q['actual_quiescent_CPU_consumer']and consumer['passed']and consumer['actual_qualified_reset_and_thread_methods_consumed']and consumer['Native_ctx_high64_not_truncated']
 cases={x['case']:x for x in consumer['cases']};assert set(cases)=={'normal0','normal1','invalid_mode','entry_reset_failure','body_error','forward_failure','cleanup_reset_failure'}and all(x['passed']for x in cases.values())
 for name in ['normal0','normal1']:
  events=cases[name]['events'];assert events.index('eight_owner_reset_sync_wait_done')<events.index('arm_after_real_boundary')<events.index('actual_typed_forward')<events.index('eight_owner_forward_completed_and_CPU_copy')<events.index('read0')and events[-1]=='clear_after_real_boundary'
 assert cases['invalid_mode']['events']==[]and cases['entry_reset_failure']['events']==['actual_typed_reset']
 assert cases['cleanup_reset_failure']['events'][-1]=='actual_typed_reset'and cases['cleanup_reset_failure']['actual_reset_calls']==2
 assert q['library_SHA256']=='e3e39de9524af7c19f6bf757236e3288c4bd7647909b2ad14f3e3ebf6d1426e9'and q['Model_Runtime_constructor_executed']is False and q['actual_Model_forward_executed']is False
 return dict(passed=True,scope='CPU compilation/ELF real original-class linkage and new control consumer only; no Model/Runtime/device execution',actual_CPU='910407.902',Slurm='COMPLETED0/21s',archive_SHA256=ARCHIVE,receipt_SHA256=RECEIPT,baseline_CPU889_receipt_SHA256=BASE_RECEIPT,raw_files=11,original_Native_objects=48,replaced_HOST=2,unchanged_objects=46,reused_HIP=hip,final_Native_objects=49,pure_scope_HOST=1,GPU_compiles=0,all_old_strong_exports_retained=len(strong),all_old_C_exports_retained=len(Cexports),nonpublic_weak_removed_diagnostic=missing,new_CABI=required,actual_unchanged_FFN_original_MoE_types_resolved=sorted(required_types),DT_NEEDED_unchanged=True,default_unarmed_OFF=True,actual_scope_CPU_8thread=True,actual_new_request_consumer_7cases=True,real_Native_reset_forward_completion_are_recording_leaves=True,quiescence_rule='Actual owner reset+wait/drain or original successful synchronous complete forward, never local active=false/HostReturned',library=dict(path=q['library'],SHA256=q['library_SHA256'],binary_SHA_from_actual_CPU_producer_not_locally_rehashed=True),Model_Runtime_constructor_executed=False,Model_forward_executed=False,GPU_GO=False,Model_GO=False,Model_gain='UNKNOWN',limits=['Some46reused objects have no adjacentMD, not49freshMD; exact original frozen objects retained','CPU uses actual scope header and original reset/check_thread methods with explicit Native execution/weights leaves','Future private Model loader must consume new Native identity/49object receipt and route planned requests through QuiescentMode','nonT1 shared/bank fallback calls may include routed experts; T1actualhits and totalMoEentries must be checked using real forward geometry','Original qualified public afc6/HC0 provider remains unchanged; not a Model/device readiness claim'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('baseline');p.add_argument('--output');a=p.parse_args();q=analyze(a.archive,a.baseline);text=json.dumps(q,indent=2)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf8')
 else:print(text)
