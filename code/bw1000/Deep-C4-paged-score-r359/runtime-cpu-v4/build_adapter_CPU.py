"""Compile only new whole-transaction C ABI; reuse CPU714 private objects."""
from pathlib import Path
import hashlib,json,subprocess,os,time,shlex,traceback
R=Path('/refactor');S=Path(__file__).resolve().parent
P=R/'incoming/Deep-C4-paged-score-r359-source-cpu-v2'
OLD=R/'candidates/Deep-C4-paged-score-r359-source-cpu-v2'
B=R/'candidates/Deep-C4-paged-score-r359-runtime-cpu-v4';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
def run(args,label):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=180)
 (B/(label+'.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,(label,p.returncode,p.stderr[-4000:]);return p.stdout
try:
 assert sha(R/'candidates/Deep-C4-paged-score-r359-env-check-v3/CPU-env-check.json')=='eb9e6aa7bc46bc6159aa24f583e62750b21fa3084399fbede46c3717085ebf40'
 prior=json.loads((R/'candidates/Deep-C4-paged-score-r359-env-check-v3/CPU-env-check.json').read_bytes());assert prior['completed']
 for n,h in prior['artifacts_after_SHA256'].items():assert sha(OLD/n)==h
 assert sha(P/'SOURCE_MANIFEST.json')=='38f8a95cfa8c6c7669ea43f66d1cb7d9507182a167feae23e2e93e1c762fc4ae'
 m=json.loads((P/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(P/n)==h for n,h in m['files_SHA256'].items())
 entry=json.loads((R/'candidates/deep-c4-smallt-r9252/CPU-build.json').read_bytes())['compiled'][0]
 assert hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='cae8b90d13beb79a2264d46a6e1016cb005dba3268affa39db0128294ce0c9d8'
 for p,h in entry['deps'].items():assert sha(p)==h,p
 args=entry['compile_argv'][:];args[1:1]=['-I'+str(S),'-I'+str(P)]
 args[args.index('-c')+1]=str(S/'adapter.cpp');args[args.index('-o')+1]=str(B/'adapter.o');args[args.index('-MF')+1]=str(B/'adapter.d')
 run(args,'adapter-compile')
 fresh=[v for v in args if v!='-MD'];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M'
 parse=lambda s:set(shlex.split(s.replace(chr(92)+chr(10),' ').split(':',1)[1]))
 deps=parse((B/'adapter.d').read_text());assert deps==parse(run(fresh,'fresh-M'))
 assert any('/Model/Deep/InfiniLM/csrc/' in p and p.endswith('owner_affinity_r2392.hpp')for p in deps)
 hpp=R/'candidates/deep-c4-projection-model-r5920/incoming/deepseek_v4_c4_indexer.hpp'
 assert str(hpp)in deps and sha(hpp)=='b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6'
 assert str(P/'private_indexer_r359.hpp')in deps
 receipt=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(receipt)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5'
 native=json.loads(receipt.read_bytes())['families']['Deep']['Native'];assert sha(native['new'])=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 libs=[v for v in native['link_argv']if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic']
 lib=B/'libDeep_C4_runtime_r359.so'
 run(['g++',str(B/'adapter.o'),str(OLD/'private.o'),str(OLD/'score.o'),native['new'],'-shared','-fPIC',*libs,'-L/opt/dtk/lib','-lamdhip64','-Wl,--no-undefined','-Wl,-rpath,'+str(Path(native['new']).parent),'-o',str(lib)],'link')
 assert 'not found'not in run(['ldd',str(lib)],'ldd')
 nm=run(['nm','-D','--defined-only',str(lib)],'nm')
 names=['error','id_size','id','create','keys','load','finish','reset','forward','proof_info','proof_read','stats','destroy','shutdown']
 assert all('deep_c4_fixture_'+n+'_r359' in nm for n in names)
 assert 'deep_paged_scores_r359' in nm
 q.update(completed=True,scope='Original afc6 C4 versus private paged C4 whole prepare/commit adapter compile/link only; no Runtime created',MD_equals_fresh_M=True,dependencies_SHA256={p:sha(p)for p in deps},library=str(lib),library_SHA256=sha(lib),reused_artifacts=prior['artifacts_after_SHA256'],new_object_SHA256=sha(B/'adapter.o'),CAPI=names,GPU_GO=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-start
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_CPU','error','seconds']}));assert q['completed']
