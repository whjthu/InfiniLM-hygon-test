"""Exactly two HOST compile commands; no Native dlopen/Runtime/weights/GPU."""
from pathlib import Path
import json,hashlib,os,subprocess,shlex,time,traceback,importlib.util
R=Path('/refactor');S=Path(__file__).resolve().parent;B=R/'candidates/Deep-C4-phase-enclosures-r413-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,Runtime_created=False,weights_opened=False,Model_loaded=False,new_HIP_compile=False,
 actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['DEEP406_PUBLICATION_COMMIT'],commands=[])
started=time.monotonic()
def run(args,label,timeout=90):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-4500:],p.stdout[-1000:]);return p.stdout
def deps(t):return set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]))
try:
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP406_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 pins=json.loads((S/'PARENT_PINS.json').read_bytes());assert sha(pins['current_976f_CPU_receipt'])==pins['current_976f_CPU_receipt_SHA256'];current=json.loads(Path(pins['current_976f_CPU_receipt']).read_bytes());assert current['passed']
 assert len(current['all_objects_SHA256'])==49 and all(sha(p)==h for p,h in current['all_objects_SHA256'].items())
 assert sha(pins['current_976f_library'])==pins['current_976f_library_SHA256']==current['library_SHA256']
 old=json.loads(Path(pins['old_CPU723_receipt']).read_bytes());assert sha(pins['old_CPU723_receipt'])==pins['old_CPU723_receipt_SHA256']and old['completed']
 entry=json.loads((R/'candidates/deep-c4-smallt-r9252/CPU-build.json').read_bytes())['compiled'][0]
 assert sha(entry['source'])==pins['current_C4_source_SHA256']and sha(pins['current_C4_object'])==pins['current_C4_object_SHA256']
 import rebase_source
 original=Path(entry['source']).read_text();original_header=R/'candidates/deep-tailSUM-r7212/incoming/deepseek_v4_c4_indexer.hpp';assert sha(original_header)==rebase_source.HEADER_SHA
 private,header=rebase_source.rebase(original,original_header.read_text());assert private.encode()==(S/'private_indexer.cpp').read_bytes()and header.encode()==(S/'private_indexer.hpp').read_bytes()
 history=dict(old['dependencies_SHA256']);history.update(entry['deps']);assert all(sha(p)==h for p,h in history.items())
 objects=[];all_deps={}
 for name in ['private_indexer','adapter']:
  args=entry['compile_argv'][:];args[1:1]=['-I'+str(S),'-I'+str(Path(entry['source']).parent),'-ffunction-sections','-fdata-sections','-pthread'];args[args.index('-c')+1]=str(S/(name+'.cpp'));args[args.index('-o')+1]=str(B/(name+'.o'));args[args.index('-MF')+1]=str(B/(name+'.d'))
  fresh=[v for v in args if v!='-MD'];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M'
  before={p:sha(p)for p in deps(run(fresh,name+'-fresh-M'))}
  for p,h in before.items():
   if Path(p).is_relative_to(S):continue
   if p in history:assert h==history[p],p
   else:assert not p.startswith('/refactor/'),('Unknown project dependency',p)
  run(args,name+'-compile',105);assert deps((B/(name+'.d')).read_text())==set(before)and all(sha(p)==h for p,h in before.items());all_deps.update(before);objects.append(str(B/(name+'.o')))
 # Compile adapter once; GC-link only its HOST main for the actual timer/TLS contract.
 exe=B/'CPU-owner-timer-contract';run(['g++',objects[1],'-Wl,--gc-sections','-pthread','-o',str(exe)],'CPU-contract-link',25)
 exe_ld=run(['ldd',str(exe)],'CPU-contract-ldd',15);assert not any(s in exe_ld.lower()for s in ['infinilm','infiniops','cudart','amdhip','infinirt','infiniccl'])
 timer=json.loads(run([str(exe)],'CPU-actual-owner-timer',20).splitlines()[-1]);assert timer['passed']
 consumers=json.loads(run(['python3','-B',str(S/'check_CPU.py')],'CPU-whole-driver',30).splitlines()[-1]);assert consumers['passed']
 link=pins['link_argv_old_adapter'][:];link[1]=objects[1];link[link.index('/refactor/candidates/Deep-C4-paged-score-r359-source-cpu-v1/private.o')]=objects[0];link.remove(pins['old_HIP_object'])
 link=[pins['current_976f_library']if p=='/refactor/candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'else '-Wl,-rpath,'+str(Path(pins['current_976f_library']).parent)if p=='-Wl,-rpath,/refactor/candidates/latest-family-native-r269/Deep'else p for p in link]
 library=B/'libDeep_C4_phase_enclosure_r413.so';link[link.index('-o')+1]=str(library);link.extend(['-Wl,-Bsymbolic','-Wl,-soname,libDeep_C4_phase_enclosure_r413.so','-pthread']);run(link,'link',40)
 ld=run(['ldd','-r',str(library)],'ldd-r',30);assert 'not found'not in ld and'undefined symbol'not in ld
 symbols=run(['nm','-D','--defined-only',str(library)],'nm',20);assert all('deep_c4_phases_'+n+'_r413'in symbols for n in ['error','create','forward','stats','destroy','shutdown'])
 assert 'phase_enclosure_r413'in symbols and 'modern_deep_allselected_set_r813'not in symbols and 'modern_deep_c4_smallt_mode_r9252'not in symbols
 dynamic=run(['readelf','-d',str(library)],'dynamic',15);needed=[l.split('[',1)[1].split(']',1)[0]for l in dynamic.splitlines()if'NEEDED'in l]
 assert any('libDeep_HC_T1_Model_r366.so'in n for n in needed)and not any('libDeep_latest_native_r269' in n for n in needed)
 for p,v in pins['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),library=str(library),library_SHA256=sha(library),HOST_compile_count=2,objects_SHA256={p:sha(p)for p in objects},dependencies_SHA256=all_deps,MD_equals_fresh_M=True,current_Native_SHA256=sha(pins['current_976f_library']),Native49objects_unchanged=True,SDK_bindings=pins['actual_DTK_bindings'],DT_NEEDED=needed,private_namespace=True,Bsymbolic=True,actual_CPU_timer_contract=timer,actual_CPU_whole_driver=consumers,GPU_GO=False,Model_gain='UNKNOWN',performance_trials=0)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-started;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','completed','actual_CPU','seconds','error']}));assert q['passed']
