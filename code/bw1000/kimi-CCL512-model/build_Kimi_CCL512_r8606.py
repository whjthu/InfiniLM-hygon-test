"""One Delta TU only, no CPU dlopen/GPU; actual parent linked objects reused."""
import pathlib,json,hashlib,subprocess,shlex,os,datetime
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-CCL512-r8606';b.mkdir(exist_ok=False);(b/'src').mkdir()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out=dict(completed=False,CPU_only=True,GPU=False,dlopen=False,commands=[],compiled_units=[])
def run(cmd,label):
 out['commands'].append(cmd)
 with(b/(label+'.log')).open('x')as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180)
 assert p.returncode==0,(label,(b/(label+'.log')).read_text()[-6000:])
try:
 old=json.loads((r/'results/Kimi-prefill512-CPU-r8564.json').read_text());base=json.loads((r/'results/Kimi-head-Model-CPU-r8344.json').read_text())
 assert old['completed']and sha(old['library'])==old['library_SHA256']=='e3e335857563db06e12e8a06cda8f8e149b8d46aebf45b7ee51a4dba1a554753'
 for p,h in old['all_linked_object_SHA256'].items():assert sha(p)==h,p
 for p,h in old['actual_dynamic_dependency_SHA256'].items():assert sha(p)==h,p
 targets=[]
 for p in old['all_linked_object_SHA256']:
  symbols=subprocess.check_output(['nm','-C','--defined-only',p],text=True)
  if any(' T 'in line and 'KimiK3DeltaAttention::forward('in line for line in symbols.splitlines()):targets.append(p)
 assert len(targets)==1,targets
 source=r/'incoming/kimi_k3_delta_attention_CCL512_r8603.cpp';proof=json.loads((r/'incoming/Kimi-CCL512-CPU-source-r8606.json').read_text());assert sha(source)==proof['source_SHA256'][source.name]
 local=b/'src'/source.name;local.write_bytes(source.read_bytes())
 helper=r/'incoming/r8564_kimi_spec_math_r3676.hpp';assert sha(helper)==proof['source_SHA256'][helper.name];(b/'src/kimi_spec_math_r3676.hpp').write_bytes(helper.read_bytes())
 flags=base['commands'][0][:base['commands'][0].index('-MD')]
 actual='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc/models/kimi_k3'
 flags.insert(6,'-I'+actual)
 obj=b/'delta.o';md=b/'delta.d';run([*flags,'-MD','-MF',str(md),'-c',str(local),'-o',str(obj)],'delta')
 deps=sorted(set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1])))
 critical={p:sha(p)for p in deps if pathlib.Path(p).name in ['kimi_k3_trace.hpp','kimi_k3_kda_trace.hpp','runtime.hpp','forward_context.hpp','tensor.hpp','kimi_k3_delta_attention.hpp']}
 assert critical and all('/incoming/'not in p for p in critical),critical
 out['compiled_units']=[dict(source=str(local),source_SHA256=sha(local),object=str(obj),object_SHA256=sha(obj),deps={p:sha(p)for p in deps},critical_header_SHA256=critical)]
 link=list(old['commands'][-1]);assert link.count(targets[0])==1;link[link.index(targets[0])]=str(obj)
 library=b/'libKimi_CCL512_r8606.so';link[link.index('-o')+1]=str(library)
 link=[('-Wl,-Map,'+str(b/'link.map'))if x.startswith('-Wl,-Map,')else x for x in link];run(link,'link')
 defined=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
 assert 'modern_kimi_prefill512_counts_r8562'in defined and 'modern_kimi_CCL512_counts_r8603'in defined
 out.update(completed=True,step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],library=str(library),library_SHA256=sha(library),parent_DSO_SHA256=old['library_SHA256'],replaced_object=targets[0],all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},actual_dynamic_dependency_SHA256=old['actual_dynamic_dependency_SHA256'],nm_defined=defined)
except BaseException as error:out['error']=repr(error);raise
finally:
 p=r/'results/Kimi-CCL512-CPU-r8606.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:out.get(k)for k in ['completed','step','error','library','library_SHA256']}))
