"""One HOST adapter compile/link, old HIP object reused, zero Native dlopen."""
from pathlib import Path
import hashlib,json,os,subprocess,time,shlex,traceback
R=Path('/refactor');S=Path(__file__).resolve().parent;B=R/'candidates/Deep-C4-same-cache-numeric-r390-cpu-v1';B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,CPU_only=True,GPU_initialized=False,Native_dlopen=False,Runtime_created=False,weights_opened=False,Model_loaded=False,new_HIP_compile=False,
 actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],publication_commit=os.environ['DEEP390_PUBLICATION_COMMIT'],commands=[])
started=time.monotonic()
def run(args,label,timeout=100):
 q['commands'].append(args);p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);(B/(label+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,(label,p.stderr[-5000:],p.stdout[-1500:]);return p.stdout
def deps(t):return set(shlex.split(t.replace('\\\n',' ').split(':',1)[1]))
try:
 mp=S/'SOURCE_MANIFEST.json';assert sha(mp)==os.environ['DEEP390_CPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert all(sha(S/n)==h for n,h in m['files_SHA256'].items())
 pins=json.loads((S/'PARENT_PINS.json').read_bytes());assert sha(pins['old_CPU723_receipt'])==pins['old_CPU723_receipt_SHA256'];old=json.loads(Path(pins['old_CPU723_receipt']).read_bytes());assert old['completed']
 assert sha(pins['old_a4e_DSO'])==pins['old_a4e_DSO_SHA256'];assert sha(pins['old_HIP_object'])==pins['old_HIP_object_SHA256']
 cp=Path(pins['current_976f_CPU_receipt']);assert sha(cp)==pins['current_976f_CPU_receipt_SHA256'];current=json.loads(cp.read_bytes());assert current['passed']and current['CPU_only']and not current['GPU_initialized']and not current['Native_dlopen']
 assert sha(pins['current_976f_library'])==pins['current_976f_library_SHA256']==current['library_SHA256']
 assert current['all_objects_SHA256'][pins['current_C4_object']]==pins['current_C4_object_SHA256']==sha(pins['current_C4_object'])
 for p,h in current['all_objects_SHA256'].items():assert sha(p)==h
 entry=json.loads((R/'candidates/deep-c4-smallt-r9252/CPU-build.json').read_bytes())['compiled'][0]
 assert hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='cae8b90d13beb79a2264d46a6e1016cb005dba3268affa39db0128294ce0c9d8'
 assert sha(entry['source'])==pins['current_C4_source_SHA256'];original=Path(entry['source']).read_text()
 assert 'pending.owner=this;pending.revision=revision_'in original and 'pending.owner!=this||pending.revision!=revision_'in original
 history=dict(old['dependencies_SHA256']);history.update(entry['deps']);assert all(sha(p)==h for p,h in history.items())
 args=entry['compile_argv'][:];args[1:1]=['-I'+str(S)];args[args.index('-c')+1]=str(S/'adapter.cpp');args[args.index('-o')+1]=str(B/'adapter.o');args[args.index('-MF')+1]=str(B/'adapter.d')
 fresh=[v for v in args if v!='-MD'];i=fresh.index('-MF');del fresh[i:i+2];i=fresh.index('-o');del fresh[i:i+2];fresh[fresh.index('-c')]='-M'
 before={p:sha(p)for p in deps(run(fresh,'fresh-M'))}
 for p,h in before.items():
  if Path(p).is_relative_to(S):continue
  if p in history:assert h==history[p],p
  else:assert not p.startswith('/refactor/'),('Unknownprojectdependency',p)
 headers={'deepseek_v4_c4_indexer.hpp':'b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6','deepseek_v4_temporal_compressor.hpp':'1b5de31aa0b27e28bb96f41cf68ee52ee58a4784e5ea3bf5e347b0c6e22b157a','deep_reduce_wide_r3443.hpp':'72eaf08c601297e45d5011c44ef1305fc290a7a73b6129cd919aa1ce6d61c2e7'}
 for name,h in headers.items():assert len([p for p,v in before.items()if Path(p).name==name and v==h])==1,(name,h)
 run(args,'adapter-compile');assert deps((B/'adapter.d').read_text())==set(before)and all(sha(p)==h for p,h in before.items())
 link=pins['link_argv_old_adapter'][:];link[1]=str(B/'adapter.o');link.remove('/refactor/candidates/Deep-C4-paged-score-r359-source-cpu-v2/private.o')
 link=[pins['current_976f_library']if p=='/refactor/candidates/latest-family-native-r269/Deep/libDeep_latest_native_r269.so'else '-Wl,-rpath,'+str(Path(pins['current_976f_library']).parent)if p=='-Wl,-rpath,/refactor/candidates/latest-family-native-r269/Deep'else p for p in link]
 library=B/'libDeep_C4_same_cache_r390.so';link[link.index('-o')+1]=str(library);assert [p for p in link if p.endswith('.o')]==[str(B/'adapter.o'),pins['old_HIP_object']];run(link,'link',60)
 ld=run(['ldd','-r',str(library)],'ldd-r',45);assert 'not found'not in ld and'undefined symbol'not in ld
 symbols=run(['nm','-D','--defined-only',str(library)],'nm');names=['error','id_size','id','create','keys','load','finish','reset','prefix','prepare','score','commit','identity','proof_info','proof_read','destroy','shutdown']
 assert all('deep_c4_lease_'+name+'_r390'in symbols for name in names)and'deep_paged_scores_r359'in symbols
 dynamic=run(['readelf','-d',str(library)],'dynamic');needed=[l.split('[',1)[1].split(']',1)[0]for l in dynamic.splitlines()if'NEEDED'in l]
 assert any('libDeep_HC_T1_Model_r366.so'in n for n in needed)and not any('libDeep_latest_native_r269.so'in n for n in needed)
 for p,v in pins['actual_DTK_bindings'].items():assert sha(p)==v['SHA256']
 assert sha(pins['existing_HIP_provider']['path'])==pins['existing_HIP_provider']['SHA256']
 exe=B/'CPU-lease-contract';run(['g++','-std=c++17','-O0','-UNDEBUG','-I'+str(S),str(S/'CPU_lease_contract.cpp'),'-o',str(exe)],'CPU-lease-compile',30);lease=json.loads(run([str(exe)],'CPU-lease-contract',10));assert lease['passed']
 consumers={}
 for name in ['check_consumer_CPU.py','check_interface_CPU.py']:
  output=run(['python3','-B',str(S/name)],name,45);result=json.loads(output.splitlines()[-1]);assert result['passed'];consumers[name]=result
 q.update(completed=True,passed=True,source_manifest_SHA256=sha(mp),library=str(library),library_SHA256=sha(library),new_HOST_object_SHA256=sha(B/'adapter.o'),old_HIP_object=pins['old_HIP_object'],old_HIP_object_SHA256=sha(pins['old_HIP_object']),current_Native_SHA256=sha(pins['current_976f_library']),MD_equals_fresh_M=True,dependencies_SHA256=before,original_header_contracts=headers,link_argv=link,DT_NEEDED=needed,actual_DTK_bindings=pins['actual_DTK_bindings'],CPU_actual_lease_protocol=lease,CPU_actual_consumers=consumers,GPU_GO=False,actual_samecache_GPU_and_nextstep_state='NOT_RUN',Model_gain='UNKNOWN',performance_trials=0)
except BaseException as exc:q.update(error=repr(exc),traceback=traceback.format_exc())
q['seconds']=time.monotonic()-started;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','seconds','error']}));assert q['passed']
