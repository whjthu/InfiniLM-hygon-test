"""Reuse CPU721 adapter object; validate actual headers and link without Runtime init."""
from pathlib import Path
import json,hashlib,os,subprocess,time,traceback,shlex
R=Path('/refactor');S=Path(__file__).resolve().parent
A=R/'candidates/Deep-C4-paged-score-r359-runtime-cpu-v4'
P=R/'candidates/Deep-C4-paged-score-r359-source-cpu-v2'
B=R/'candidates/Deep-C4-paged-score-r359-runtime-link-v5';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,GPU_initialized=False,Native_dlopen=False,recompiled=False,commands=[]);start=time.monotonic()
def run(args,label):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=90);(B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-3000:]);return p.stdout
try:
 assert sha(A/'CPU-build.json')=='79d982c7bb8e3e67bc72f5b74bb6310aa95d73ec3410d800fec924450eeeea5d'
 old=json.loads((A/'CPU-build.json').read_bytes());assert old['completed']is False and 'owner_affinity_r2392.hpp' in old['traceback']
 assert sha(A/'adapter.o')=='94fa580f3a87626046991e7d62c8a0e97445006399f8844a2c4be5fb8e3daacb'
 src=R/'incoming/Deep-C4-paged-score-r359-runtime-cpu-v4';assert sha(src/'SOURCE_MANIFEST.json')=='eb686da91ac6c6da1ca1c0af65a8144d4a06acee789fb451839cdc8c3ae2153e'
 m=json.loads((src/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(src/n)==h for n,h in m['files_SHA256'].items())
 parse=lambda s:set(shlex.split(s.replace(chr(92)+chr(10),' ').split(':',1)[1]))
 deps=parse((A/'adapter.d').read_text());assert deps==parse((A/'fresh-M.log').read_text())
 header=R/'candidates/deep-tailSUM-r7212/incoming/deepseek_v4_c4_indexer.hpp'
 assert str(header)in deps and sha(header)=='b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6'
 assert str(R/'incoming/Deep-C4-paged-score-r359-source-cpu-v2/private_indexer_r359.hpp')in deps
 assert not any(p.endswith('/owner_affinity_r2392.hpp')for p in deps)
 entry=json.loads((R/'candidates/deep-c4-smallt-r9252/CPU-build.json').read_bytes())['compiled'][0]
 assert hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='cae8b90d13beb79a2264d46a6e1016cb005dba3268affa39db0128294ce0c9d8'
 for p,h in entry['deps'].items():assert sha(p)==h,p
 env=R/'candidates/Deep-C4-paged-score-r359-env-check-v3/CPU-env-check.json';assert sha(env)=='eb9e6aa7bc46bc6159aa24f583e62750b21fa3084399fbede46c3717085ebf40'
 prior=json.loads(env.read_bytes());assert prior['completed']
 for n,h in prior['artifacts_after_SHA256'].items():assert sha(P/n)==h,n
 receipt=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(receipt)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5'
 native=json.loads(receipt.read_bytes())['families']['Deep']['Native'];assert sha(native['new'])=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 libs=[v for v in native['link_argv']if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic'];lib=B/'libDeep_C4_runtime_r359.so'
 run(['g++',str(A/'adapter.o'),str(P/'private.o'),str(P/'score.o'),native['new'],'-shared','-fPIC',*libs,'-L/opt/dtk/lib','-lamdhip64','-Wl,--no-undefined','-Wl,-rpath,'+str(Path(native['new']).parent),'-o',str(lib)],'link')
 assert 'not found'not in run(['ldd',str(lib)],'ldd')
 nm=run(['nm','-D','--defined-only',str(lib)],'nm');names=['error','id_size','id','create','keys','load','finish','reset','forward','proof_info','proof_read','stats','destroy','shutdown']
 assert all('deep_c4_fixture_'+n+'_r359' in nm for n in names)and 'deep_paged_scores_r359'in nm
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),adapter_object_SHA256=sha(A/'adapter.o'),reused_artifacts=prior['artifacts_after_SHA256'],dependencies_SHA256={p:sha(p)for p in deps},MD_equals_original_fresh_M=True,original_C4_header_MD_path=str(header),original_C4_header_SHA256=sha(header),owner_affinity='No submit/apply_once inline in adapter TU; Runtime::create is external pinned afc6. Actual eight owner thread mask check required on GPU, not qualified by adapter compile',CPU721_failure_preserved=True,GPU_GO=False,Runtime_adapter='Compiled/linked, not executed',GPU_driver='PENDING')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start
(B/'CPU-link.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_CPU','error','seconds']}));assert q['completed']
