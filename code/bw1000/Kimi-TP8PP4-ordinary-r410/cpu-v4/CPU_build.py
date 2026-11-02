"""One head HOST object only. Reuse CPU823 policy and CPU819 three objects."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time
D=Path(__file__).resolve().parent;N=D/'native';R=Path('/refactor')
B=R/'candidates/Kimi-TP8PP4-ordinary-r410-CPU-v4'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert 'torch'not in sys.modules and not os.environ.get('LD_PRELOAD')
 mf=D/'CPU_SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());pins=json.loads((D/'PARENT.json').read_bytes())
 for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
 assert not B.exists();B.mkdir();started=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=0,Native_dlopen=False,Runtime_created=False,GPU_GO=False,new_HOST_TUs=1,new_GPU_TUs=0,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(mf),logs={},commands=[])
 def run(a,label,limit=90):
  left=270-(time.monotonic()-started);assert left>0
  q['commands'].append(a);p=subprocess.run(a,capture_output=True,text=True,timeout=min(left,limit));log=B/(label+'.log');log.write_text(p.stdout+p.stderr);q['logs'][label]=dict(rc=p.returncode,SHA256=sha(log))
  if p.returncode:raise RuntimeError(label+' failed: '+str(log))
  return p.stdout
 def receipt(p,h):
  assert sha(p)==h,p;v=json.loads(Path(p).read_bytes());assert v['completed']and v['CPU_only'];return v
 try:
  parent=receipt(R/'candidates/Kimi-TP8PP4-ordinary-r410-CPU-v2/CPU-result.json',pins['parent_CPU823_receipt_SHA256'])
  assert parent['passed']and sha(parent['native'])==parent['native_SHA256']==pins['parent_Native_SHA256']
  old=receipt(R/'candidates/Kimi-shard-only-Model-r9183/CPU-build.json',pins['head_parent_receipt_SHA256'])
  for p,h in parent['all_linked_object_SHA256'].items():assert sha(p)==h,p
  for v in parent['actual_dependency_library_SHA256'].values():assert sha(v['path'])==v['SHA256']
  header='/refactor/incoming/Kimi-shard-only-Model-r9183/kimi_shard_stats_r9183.hpp'
  assert sha(header)==pins['original_head_header_SHA256']
  matches=[(p,h)for p,h in parent['HOST']['policy']['all_new_MD'].items()if Path(p).name=='kimi_shard_stats_r9183.hpp'];assert len(matches)==1 and matches[0][1]==sha(N/'kimi_shard_stats_r9183.hpp')
  # The original r9183 introduction compiled exactly these two consumers.
  assert set(old['new_complete_freshMD'])=={'head','policy'}
  assert all(header in v and v[header]==pins['original_head_header_SHA256']for v in old['new_complete_freshMD'].values())
  replacements={};details={};contract_flags=None
  for unit,name in [('head','kimi_shard_model_r9183.cpp')]:
   original=next(a[:]for a in old['commands']if '-c'in a and Path(a[a.index('-c')+1]).name==name)
   flags=[];i=0
   while i<len(original):
    a=original[i]
    if a in ['-o','-MF','-MT','-c']:i+=2;continue
    if a in ['-MD','-MMD','-MP']:i+=1;continue
    flags.append(a);i+=1
   flags.insert(1,'-I'+str(N));contract_flags=flags
   out=B/(unit+'.o');run([*flags,'-MD','-MF',str(B/(unit+'.d')),'-c',str(N/name),'-o',str(out)],unit+'-HOST')
   run([*flags,'-M','-MF',str(B/(unit+'.fresh.d')),'-MT',str(out),str(N/name)],unit+'-fresh-M',30)
   actual=deps(B/(unit+'.d'));assert actual==deps(B/(unit+'.fresh.d'))
   parentdir=Path('/refactor/incoming/Kimi-shard-only-Model-r9183')
   normalized={str(parentdir/Path(p).name)if Path(p).parent==N else p:p for p in actual}
   assert set(normalized)==set(old['new_complete_freshMD'][unit]),(unit,'MD consumer set changed')
   for p,real in normalized.items():
    if p not in {header,str(parentdir/'kimi_shard_head_r9183.hpp')}:assert sha(real)==old['new_complete_freshMD'][unit][p],(unit,p)
   oldobj='/refactor/candidates/Kimi-TP8PP4-ordinary-r410-CPU-v2/'+unit+'.o';assert sha(oldobj)==parent['all_linked_object_SHA256'][oldobj]
   replacements[oldobj]=str(out);details[unit]=dict(source_SHA256=sha(N/name),object_SHA256=sha(out),all_new_MD={p:sha(p)for p in sorted(actual)},fresh_MD_matches=True)
  # Same successful link argv lineage; use exact CPU819 object order.
  link=next(a[:]for a in old['commands']if '-shared'in a)
  object_order=list(parent['all_linked_object_SHA256']);cursor=0
  for i,p in enumerate(link):
   if p.endswith('.o'):link[i]=object_order[cursor];cursor+=1
  assert cursor==33 and {p for p in link if p.endswith('.o')}==set(parent['all_linked_object_SHA256'])
  link=[replacements.get(p,p)for p in link];native=B/'libKimi_TP8PP4_ordinary_r410.so';link[link.index('-o')+1]=str(native)
  link=[('-Wl,-Map,'+str(B/'link.map')if p.startswith('-Wl,-Map')else p)for p in link]
  run(link,'Native-link')
  symbols=lambda s:{l.split()[-1]for l in s.splitlines()if l.split()and l.split()[-1].startswith('modern_')}
  previous=symbols(run(['nm','-D','--defined-only',parent['native']],'parent-CAPI',20));current=symbols(run(['nm','-D','--defined-only',str(native)],'current-CAPI',20));assert previous==current and len(current)==133
  ldd=run(['ldd','-r',str(native)],'ldd-r',45);assert 'not found'not in ldd and 'undefined symbol:'not in ldd
  resolved={}
  for line in ldd.splitlines():
   w=line.split();p=(w[2]if len(w)>2 and w[1]=='=>'else w[0]if w and w[0].startswith('/')else None)
   if p and Path(p).name in parent['actual_dependency_library_SHA256']:
    key=Path(p).name;assert sha(p)==parent['actual_dependency_library_SHA256'][key]['SHA256'];resolved[key]=dict(path=p,SHA256=sha(p))
  assert resolved==parent['actual_dependency_library_SHA256']
  exe=B/'CPU-contract';run([*contract_flags,str(D/'CPU_contract.cpp'),'-o',str(exe)],'header-contract-compile',45)
  q['actual_header_contract']=json.loads(run([str(exe)],'header-contract-run',10))
  q.update(completed=True,passed=True,Native_linked=True,native=str(native),native_SHA256=sha(native),parent_CPU823_receipt_SHA256=pins['parent_CPU823_receipt_SHA256'],parent_CPU819_receipt_SHA256=pins['parent_CPU819_receipt_SHA256'],parent_Native_SHA256=parent['native_SHA256'],HOST=details,all_linked_object_SHA256={p:sha(p)for p in link if p.endswith('.o')},unchanged_objects_SHA256={p:h for p,h in parent['all_linked_object_SHA256'].items()if p not in replacements},actual_dependency_library_SHA256=resolved,retained_CAPI=len(current),MD_changed_head_header_consumers=['head'],qualification='Actual header TP8+TP16 source/descriptor checks and one HOST compile/link only; no Runtime/model/physical fit/quality')
 except BaseException as e:q['error']=repr(e);raise
 finally:
  q['elapsed_s']=time.monotonic()-started;(B/'CPU-result.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','elapsed_s','error','native_SHA256']}))
if __name__=='__main__':main()
