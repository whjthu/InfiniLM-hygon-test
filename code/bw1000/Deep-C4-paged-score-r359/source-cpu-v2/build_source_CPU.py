"""CPU-only isolated source/HIP build against exact original C4 compiler closure."""
from pathlib import Path
import json,hashlib,subprocess,time,shlex,os,traceback
R=Path('/refactor');S=Path(__file__).resolve().parent;B=R/'candidates/Deep-C4-paged-score-r359-source-cpu-v2';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[],scope='Private class/kernel compile and CPU FP8/layout oracle only; Runtime adapter/actual baseline/GPU missing')
started=time.monotonic()
def run(args,label,timeout=240):
 q['commands'].append(args);v=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(v.stdout+v.stderr);assert v.returncode==0,(label,v.returncode,v.stderr[-4000:]);return v.stdout
try:
 m=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 original=R/'candidates/deep-c4-smallt-r9252/CPU-build.json';entry=json.loads(original.read_bytes())['compiled'][0]
 assert hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='cae8b90d13beb79a2264d46a6e1016cb005dba3268affa39db0128294ce0c9d8'
 assert sha(entry['source'])=='7f2ffe0395475a98be8723bf434aa9f4944cb5336656b6d57de0f93fc6d1187e' and sha(entry['object'])=='3ea73a4acea21b2bf7cd74a55ff6b0616b995802cf915227bd939adb4a898156'
 for p,h in entry['deps'].items():assert sha(p)==h,('Changed originaldependency',p)
 args=entry['compile_argv'][:];args.insert(1,'-I'+str(S));args[args.index('-c')+1]=str(S/'private_indexer_r359.cpp');args[args.index('-o')+1]=str(B/'private.o');args[args.index('-MF')+1]=str(B/'private.d')
 run(args,'private-compile')
 fresh=[v for v in args if v not in ['-MD']];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M'
 # GCC continuation removal is explicit, not shell interpolation.
 parse=lambda text:set(shlex.split(text.replace(chr(92)+chr(10),' ').split(':',1)[1]))
 deps=parse((B/'private.d').read_text());again=parse(run(fresh,'fresh-M'));assert deps==again
 parent=json.loads((S/'PARENT_SOURCE.json').read_bytes())
 resolution=[]
 for row in parent['ABI_headers_from_actual_original_MD']:
  if Path(row['path']).name=='deepseek_v4_c4_indexer.hpp':continue
  actual=row['path']
  if actual=='/refactor/candidates/deep-c4-smallt-r9252/deep_c4_smallt_guard_r9252.hpp':
   actual='/refactor/incoming/Deep-C4-smallT-r9252/deep_c4_smallt_guard_r9252.hpp'
  assert actual in deps and sha(actual)==row['SHA256'],('Wrongcriticalheader',row['path'],actual)
  resolution.append(dict(original_path=row['path'],actual_MD_path=actual,SHA256=row['SHA256'],bytes_identical=True))
 q['critical_header_resolution']=resolution
 assert any('/Model/Deep/InfiniLM/csrc/execution/submit.hpp' in p for p in deps)
 assert any(p.endswith('/owner_affinity_r2392.hpp') for p in deps)
 run(['hipcc','-O3','-fPIC','--offload-arch=gfx936','-ffp-contract=off','-c',str(S/'paged_scores_r359.hip'),'-o',str(B/'score.o')],'kernel-compile')
 receipt=R/'candidates/latest-family-native-r269/CPU-build.json';assert sha(receipt)=='3399988b130529836f85826e912673173831e4dd06429073578949e3c6430fc5';native=json.loads(receipt.read_bytes())['families']['Deep']['Native'];assert sha(native['new'])=='afc6f58b71ef7a51657deae452da86a31afbfa6e79d36d442ccdd26d56c0e6f2'
 libs=[v for v in native['link_argv']if v.startswith(('-L','-l','-Wl,'))and v!='-Wl,-Bsymbolic'];lib=B/'libDeep_C4_private_source_r359.so'
 run(['g++',str(B/'private.o'),str(B/'score.o'),native['new'],'-shared','-fPIC',*libs,'-L/opt/dtk/lib','-lamdhip64','-Wl,--no-undefined','-Wl,-rpath,'+str(Path(native['new']).parent),'-o',str(lib)],'link')
 ld=run(['ldd',str(lib)],'ldd');assert 'not found'not in ld
 nm=run(['nm','-D','--defined-only',str(lib)],'nm');assert 'deep_paged_scores_r359'in nm and 'paged_score_r359'in nm
 ref=json.loads(run(['python','-B',str(S/'check_reference_CPU.py')],'reference',30));assert ref['passed']and ref['GPU']==0
 q.update(completed=True,source_manifest_SHA256=sha(S/'SOURCE_MANIFEST.json'),source_SHA256={p.name:sha(p)for p in S.iterdir()if p.is_file()},original_compiled_entry_SHA256='cae8b90d13beb79a2264d46a6e1016cb005dba3268affa39db0128294ce0c9d8',all_original_dependencies_unchanged=True,MD_equals_fresh_M=True,Model_submit_and_actual_owner_affinity_header=True,dependencies_SHA256={p:sha(p)for p in deps},library=str(lib),library_SHA256=sha(lib),objects_SHA256={str(p):sha(p)for p in [B/'private.o',B/'score.o']},Native_SHA256=sha(native['new']),reference=ref,seconds=time.monotonic()-started)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','actual_CPU','error','seconds']}));assert q['completed']
