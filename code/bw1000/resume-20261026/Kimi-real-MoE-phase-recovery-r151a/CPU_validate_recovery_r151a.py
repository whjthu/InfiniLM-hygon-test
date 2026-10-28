"""Validate already-built151 artifacts in the real container; NEVER compile.

Keep491 failed receipt unchanged. Only two reviewed weak JSON conversion
constructors may disappear; all other old exports, methods and C APIs remain.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shlex,subprocess,time,traceback
R=Path('/refactor');D=Path(__file__).resolve().parent;S=R/'incoming/Kimi-real-MoE-phase-r151';B=R/'candidates/Kimi-real-MoE-phase-r151';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source-manifest-sha',required=True);a=p.parse_args();manifest=D/'source-manifest-r151a.json';assert sha(manifest)==a.source_manifest_sha
for rel,h in json.loads(manifest.read_bytes())['files_SHA256'].items():assert sha(D/rel)==h
assert sha(S/'source-manifest-r151.json')=='cd0f7864a8aaffd43be9818eaeacc6eeb996c77532cf23a4acff12f2f5b14239'
failed=B/'CPU-build.json';assert sha(failed)=='ed1d6e0666652c86a47f3f74b32cc45e2cfe5a654cd8e3fe3633a21d8c95444a';f=json.loads(failed.read_bytes());assert f['actual_step']=='910407.491'and f['completed']is False and f['CPU_only']and not f['GPU']and not f['dlopen']
assert Path('/lib/x86_64-linux-gnu/libc.so.6').exists(),'Must run in original Ubuntu/DTK container; login-host ldd is invalid context'
out=B/'CPU-build-recovered-r151a.json';assert not out.exists();review=json.loads((D/'export-review-r151a.json').read_bytes());parent=json.loads((S/'parent-build-pins-r151.json').read_bytes())
old=R/'candidates/Kimi-W2-pair-r9280/libKimi_W2_pair_r9280.so';new=B/'libKimi_real_MoE_phase_r151.so';assert sha(old)==review['old_SHA256']and sha(new)==review['new_SHA256']
q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'original_failed_actual_step':'910407.491','original_failed_receipt_SHA256':sha(failed),'new_compiles':0,'new_links':0,'new_native_loads':0,'source_manifest_SHA256':sha(S/'source-manifest-r151.json'),'recovery_source_manifest_SHA256':sha(manifest)}
def call(args,env=None):
 z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=45);assert z.returncode==0,(args[0],z.returncode);return z.stdout+z.stderr
def exports(path):
 return {v[-1]:v[-2]for line in call(['nm','-D','--defined-only',str(path)]).splitlines()for v in [line.split()]if len(v)>=3}
try:
 for rel,h in json.loads((S/'source-manifest-r151.json').read_bytes())['files_SHA256'].items():assert sha(S/rel)==h
 before,after=exports(old),exports(new);missing={s:before[s]for s in before.keys()-after.keys()};added={s:after[s]for s in after.keys()-before.keys()}
 assert missing=={row['symbol']:row['type']for row in review['missing']}
 assert added=={row['symbol']:row['type']for row in review['added']}
 assert all(before[s]=='W'for s in missing),'No strong/public symbol removal allowed'
 assert all(s in after and after[s]==before[s]for s in before if s.startswith('modern_')or 'KimiK3MoE'in s)
 assert all(s in after and after[s]==before[s]for s in before if before[s]=='T')
 assert after['modern_kimi_moe_phase_mark_r151']=='T'
 demangled={s:call(['c++filt',s]).strip()for s in missing};assert all('basic_json<std::vector<int,'in text and 'std::vector<int,'in text for text in demangled.values())
 link=f['commands'][-1];assert link[0]=='g++'and str(new)==link[link.index('-o')+1]
 objects=[x for x in link if x.endswith('.o')];replaced={str(R/'candidates/Kimi-I192-batch-Model-r9129/moe.o'):str(B/'moe.o'),str(R/'candidates/Kimi-W2-pair-r9280/provider.o'):str(B/'provider.o')}
 assert set(objects)=={replaced.get(x,x)for x in parent['all_linked_objects']}
 for path,h in parent['all_linked_objects'].items():
  if path not in replaced:assert sha(path)==h
 md={}
 for path in B.glob('*.d'):md[path.name]={s:sha(s)for s in shlex.split(path.read_text().replace('\\\n',' ').split(':',1)[1])if Path(s).is_file()}
 assert set(md)=={'moe.d','provider.d'}
 critical='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3/kimi_k3_moe.hpp';assert md['moe.d'][critical]==parent['critical_MoE_header_SHA256']
 assert md['moe.d'][str(S/'kimi_model_events_r8669zq.hpp')]=='2a36c755719cb723530795d57c14b2b5311a54cd0e6f737de532bee483cfffc2'
 env=dict(os.environ,LD_LIBRARY_PATH=parent['LD_LIBRARY_PATH']);ldd=call(['ldd','-r',str(new)],env)
 assert 'not found'not in ldd and 'undefined symbol:'not in ldd, 'Actual container relocation closure failed; no qualification'
 dynamic={};versions={}
 for label,path in [('old',old),('new',new)]:
  text=call(['readelf','-d',str(path)]);dynamic[label]=re.findall(r'\(NEEDED\).*?\[(.*?)\]',text)
  versions[label]=sorted(set(re.findall(r'Name:\s*(\S+)',call(['readelf','--version-info',str(path)]))))
 assert dynamic['new']==dynamic['old'],'DT_NEEDED changed; requires separate review'
 assert set(versions['new'])<=set(versions['old']),'New versioned dependency; requires separate review'
 q.update(completed=True,native=str(new),native_SHA256=sha(new),parent_native_SHA256=sha(old),all_linked_object_SHA256={x:sha(x)for x in objects},unchanged_objects={x:h for x,h in parent['all_linked_objects'].items()if x not in replaced},replaced_objects=replaced,new_MD=md,existing_model_event_object_reused=True,LD_LIBRARY_PATH=parent['LD_LIBRARY_PATH'],reviewed_missing_weak_symbols=review['missing'],missing_demangled=demangled,added_symbols=review['added'],all_strong_T_and_public_modern_and_MoE_exports_preserved=True,container_ldd_r_no_undefined=True,container_ldd_SHA256=hashlib.sha256(ldd.encode()).hexdigest(),DT_NEEDED=dynamic,versioned_requirements=versions,GPU_qualified=False,histogram='DEFERRED',new_sync=False,new_D2H=False)
except BaseException as error:q.update(error_class=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
out.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','CPU_only','GPU','actual_step','new_compiles','new_links','native_SHA256','error_class']}));assert q['completed']
