"""Run inside the pinned container in a CPU-only Slurm step. Never dlopen/GPU."""
import pathlib,json,hashlib,subprocess,shutil,shlex,os
r=pathlib.Path('/refactor');src=pathlib.Path(__file__).resolve().parent;b=r/'candidates/Kimi-live-taps-r9046';b.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
parent=json.loads((r/'candidates/Kimi-target-taps-r9022/CPU-build.json').read_text());assert parent['completed'] and sha(parent['native'])==parent['native_SHA256']
old='/refactor/candidates/Kimi-target-taps-r9022/trace.o';assert old in parent['all_linked_object_SHA256']
for p,h in parent['unchanged_object_SHA256'].items():assert sha(p)==h,p
compile_command=next(c for c in parent['commands'] if '-c'in c);flags=compile_command[:compile_command.index('-MD')]
flags+=['-I'+str(src),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3']
obj=b/'trace.o';commands=[[*flags,'-MD','-MF',str(b/'trace.d'),'-c',str(src/'kimi_live_trace_r9046.cpp'),'-o',str(obj)]]
link=list(parent['commands'][-1]);assert link.count(old)==1;link[link.index(old)]=str(obj);native=b/'libKimi_live_taps_r9046.so';link[link.index('-o')+1]=str(native);commands.append(link)
receipt={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'host_TUs':1,'commands':commands,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'source_root':str(src)}
try:
 for n,c in enumerate(commands):
  p=subprocess.run(c,capture_output=True,text=True,timeout=130);(b/f'command{n}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-6000:]
 expected=['modern_kimi_live_taps_begin_r9046','modern_kimi_live_taps_count_r9046','modern_kimi_live_taps_info_r9046','modern_kimi_live_taps_copy_r9046','modern_kimi_diag_capabilities_r9008','modern_kimi_diag_arm_r9008','modern_kimi_diag_read_r9008','modern_kimi_engine_create_pair_r9006','modern_kimi_pair_geometry_r9006','modern_kimi_pair_rows_r9006','modern_kimi_pair_prefill_r9006','modern_kimi_pair_forward_r9006','modern_kimi_pair_commit_r9006','modern_kimi_pair_reset_r9006','modern_kimi_pair_mode_r9007','modern_kimi_pair_ordinary_forward_r9007']
 symbols=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(n in symbols for n in expected)
 deps=shlex.split((b/'trace.d').read_text().replace('\\\n',' ').split(':',1)[1]);linked={n:sha(n) for n in link if n.endswith('.o')}
 receipt.update(completed=True,native=str(native),native_SHA256=sha(native),parent_native_SHA256=parent['native_SHA256'],source_SHA256={p.name:sha(p) for p in src.glob('*') if p.is_file()},replaced_object=old,unchanged_object_SHA256={p:h for p,h in parent['unchanged_object_SHA256'].items() if p!=old},all_linked_object_SHA256=linked,dependencies_SHA256={n:sha(n)for n in deps if pathlib.Path(n).is_file()},exports=expected)
except BaseException as e:receipt['error']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));assert receipt['completed']
