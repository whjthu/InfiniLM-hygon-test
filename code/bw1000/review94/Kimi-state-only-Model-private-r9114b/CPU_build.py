import pathlib,json,subprocess,hashlib,time,os,traceback,re
r=pathlib.Path('/refactor');src=r/'incoming/Kimi-state-only-Model-r9114b';b=r/'candidates/Kimi-state-only-Model-r9114b';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.monotonic();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'host_TUs':3,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 parent=json.loads((r/'candidates/Kimi-live-taps-r9046/CPU-build.json').read_bytes());assert parent['completed'] and sha(parent['native'])==parent['native_SHA256'];assert parent['native_SHA256']=='14bc32c2bede5f70a245bb6c132d94fab49172a761b173978c158581e6c833cd';assert sha(r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine/rank_worker.hpp')=='cc366867e69dc74db254ac0fbe634e70d0e3196e859bd6156f1bed62dc017c02'
 (src/'kimi_spec_math_r3676.hpp').write_bytes((r/'candidates/Kimi-CCL512-r8606/src/kimi_spec_math_r3676.hpp').read_bytes())
 commands=[];base=json.loads((r/'results/Kimi-CCL512-CPU-r8606.json').read_bytes())['commands'][0][:];prefix=['-I'+str(src),'-I'+str(src/'engine'),'-I/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine']
 def compile_cmd(old,source,name):
  c=old[:];c[1:1]=prefix;c[c.index('-c')+1]=str(src/source);c[c.index('-o')+1]=str(b/(name+'.o'));c[c.index('-MF')+1]=str(b/(name+'.d'));return c
 commands.append(compile_cmd(base,'delta.cpp','delta'));wold=json.loads((src/'worker-compile-parent.json').read_bytes());commands.append(compile_cmd(wold,'worker.cpp','worker'));pair_parent=json.loads((r/'candidates/Kimi-ordinary-pair-r9007/CPU-build.json').read_bytes());pc=next(c for c in pair_parent['commands']if '-c'in c and any('pair_probe_r9006.cpp'in a for a in c));commands.append(compile_cmd(pc,'pair_probe.cpp','pair_probe'))
 q['commands']=commands
 for n,c in enumerate(commands):
  p=subprocess.run(c,capture_output=True,text=True,timeout=140);(b/f'command{n}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-10000:]
 # Compare BSS symbol sizes from old/new headers; no executable, native load or GPU.
 layouts={}
 for label,header,flags in [('old','/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/engine/rank_worker.hpp',wold[:wold.index('-MD')]),('new',str(src/'engine/rank_worker.hpp'),[wold[0],*prefix,*wold[1:wold.index('-MD')]])]:
  p=b/('layout_'+label+'.cpp');p.write_text('#include "'+header+'"\nextern "C" {char abi_worker[sizeof(infinilm::engine::RankWorker)];char abi_input[sizeof(infinilm::engine::RankWorker::Input)];char abi_output[sizeof(infinilm::engine::RankWorker::Output)];char abi_alignment[alignof(infinilm::engine::RankWorker)];}\n');obj=b/('layout_'+label+'.o');c=[*flags,'-c',str(p),'-o',str(obj)];ret=subprocess.run(c,text=True,capture_output=True,timeout=100);(b/('layout_'+label+'.log')).write_text(ret.stdout+ret.stderr);assert ret.returncode==0,ret.stderr[-6000:];nm=subprocess.check_output(['nm','-S',str(obj)],text=True);layouts[label]={s.split()[-1]:int(s.split()[1],16)for s in nm.splitlines()if ' abi_'in s}
 assert layouts['old']==layouts['new'],layouts;q['ABI_layout']=layouts
 link=parent['commands'][-1][:];replacements={'/refactor/candidates/Kimi-CCL512-r8606/delta.o':b/'delta.o','/refactor/candidates/Kimi-Model-events-r8669zq1/kimi_rank_worker_model_events_r8669zq.o':b/'worker.o','/refactor/candidates/Kimi-ordinary-pair-r9007/pair_probe.o':b/'pair_probe.o'}
 compat=r/'candidates/Kimi-live-taps-compatible-r9112/CPU-build.json'
 if compat.exists():
  cp=json.loads(compat.read_bytes());assert cp['completed'];obj=r/'candidates/Kimi-live-taps-compatible-r9112/trace.o';assert sha(obj)=='4a8f1eb8848943babba19c2c9f2edb75af5d8b028fa7e1930ee30cc45c4c061f';replacements['/refactor/candidates/Kimi-live-taps-r9046/trace.o']=obj;q['optional_six_taps']=True;q['optional_trace_receipt_SHA256']=sha(compat)
 else:q['optional_six_taps']=False
 for a,z in replacements.items():assert link.count(a)==1;link[link.index(a)]=str(z)
 for p,h in parent['all_linked_object_SHA256'].items():
  if p not in replacements:assert sha(p)==h,p
 native=b/'libKimi_state_only_Model_r9114.so';link[link.index('-o')+1]=str(native)
 for j,a in enumerate(link):
  if a.startswith('-Wl,-Map,')or a.startswith('-Wl,-Map='):link[j]='-Wl,-Map,'+str(b/'link.map')
 q['commands'].append(link);p=subprocess.run(link,text=True,capture_output=True,timeout=140);(b/'link.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[-10000:];nm=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);required=['modern_kimi_state_mode_r9114','modern_kimi_state_replay_r9114','modern_kimi_state_stats_r9114','modern_kimi_state_capabilities_r9114','modern_kimi_pair_forward_r9006','modern_kimi_live_taps_begin_r9046'];assert all(s in nm for s in required)
 q.update(completed=True,native=str(native),native_SHA256=sha(native),source_root=str(src),source_SHA256={p.relative_to(src).as_posix():sha(p)for p in src.rglob('*')if p.is_file()},unchanged_object_SHA256={p:h for p,h in parent['all_linked_object_SHA256'].items()if p not in replacements},all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},replaced_objects={p:str(v)for p,v in replacements.items()},exports=required,elapsed_seconds=time.monotonic()-start)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
