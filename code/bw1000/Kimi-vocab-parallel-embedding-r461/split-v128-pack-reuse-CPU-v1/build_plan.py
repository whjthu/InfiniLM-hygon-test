"""Finite evidence-bound CPU compile/link adapter. No Native/SDK entry calls.

All changed-TU fresh-M closures complete before any compile; old objects are
reused only by exact SHA. Each immutable package carries this file's own bytes.
"""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,sys,time,traceback,shutil
D=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').partition(':')[2]))
def strong(text):return {v.split()[-1]:v.split()[-2]for v in text.splitlines()if len(v.split())>=3 and v.split()[-2]in ['T','D','B','R']}
def needed(text):return re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',text)
def paths(text):
 if 'not found'in text or 'undefined symbol'in text:raise RuntimeError('loader closure incomplete')
 return sorted(set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M)))

def main():
 start=time.monotonic();plan=json.loads((D/'BUILD_PLAN.json').read_bytes());B=Path(plan['candidate']);assert not B.exists(),'Existing candidate: reconcile original; never replay';B.mkdir();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],CPU_only=True,Native_dlopen=False,Runtime=False,SDK_explicit_calls=0,GPU_job=False,weights=False,Model_qualified=False,Torch_imported=False,ldd_child_initialization='UNKNOWN, qualified child-only preload where specified',new_HOST=0,new_GPU=0,commands=[],logs={},compiled={},libraries={})
 def save():
  t=B/'CPU-result.tmp';t.write_text(json.dumps(q,indent=2)+'\n');os.replace(t,B/'CPU-result.json')
 def run(args,label,env=None):
  left=265-(time.monotonic()-start);assert left>2,'Original265s CPUwork cap';q['phase']=label;q['commands'].append(dict(name=label,argv=args));save();z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=left-1,env=env)
  for k,v in [('stdout',z.stdout),('stderr',z.stderr)]:
   p=B/(label+'.'+k);p.write_text(v);q['logs'][label+'.'+k]=dict(path=str(p),bytes=p.stat().st_size,SHA256=H(p))
  save();assert z.returncode==0,(label,z.returncode,z.stderr[-1800:]);return z.stdout,z.stderr
 def subst(args,built,libs):
  values={'$D':str(D),'$B':str(B)}
  values.update({'$UNIT:'+k:v['object']for k,v in built.items()});values.update({'$LIB:'+k:v['path']for k,v in libs.items()})
  # Placeholders are exact arguments, never shell-evaluated strings.
  out=[]
  for a in args:
   if a in values:a=values[a]
   elif a.startswith('$RPATH:'):a='-Wl,-rpath,'+str(B/a[len('$RPATH:'):])
   out.append(a)
  return out
 try:
  claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert claim['CPU_GO']and not claim['GPU_GO']and H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['TUNING_CPU_CLAIM_SHA'];assert claim['parent']==os.environ['SLURM_JOB_ID'];assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256'];mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/p)==h for p,h in mf['files_SHA256'].items());q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'))
  for p,t in plan['tools'].items():assert str(Path(p).resolve())==t['canonical']and H(p)==t['sha256'],('tool identity',p)
  for k,v in plan.get('required_environment',{}).items():assert os.environ.get(k)==v,(k,'environment differs')
  q['effective_ROCM_PATH']=os.environ.get('ROCM_PATH')
  for u in plan['units']:
   compiler=shutil.which(u['prefix'][0]);assert compiler and any(str(Path(compiler).resolve())==v['canonical']and H(compiler)==v['sha256']for v in plan['tools'].values()),'compiler selection unpinned'
  for ref in plan['references']:
   assert H(ref['path'])==ref['SHA256'],ref['path']
   if ref.get('receipt'):
    v=json.loads(Path(ref['path']).read_bytes());assert v['completed']and v['passed']==ref['passed']
  for p,h in plan['objects_SHA256'].items():assert H(p)==h,('old object bytes',p)
  allowed={}
  for a in plan['dependency_maps']:
   assert H(a['path'])==a['SHA256'];v=json.loads(Path(a['path']).read_bytes())
   for k in a['keys']:v=v[k]
   for p,h in v.items():
    assert p not in allowed or allowed[p]==h,('conflicting old dep pins',p)
    allowed[p]=h
  for p,h in plan.get('extra_original_dependencies',{}).items():assert p not in allowed or allowed[p]==h;allowed[p]=h
  allowed.update({str(D/p):h for p,h in mf['files_SHA256'].items()})
  observed={};fresh={};unmatched=[]
  for unit in plan['units']:
   source=D/unit['source'];prefix=[unit['prefix'][0],'-I'+str(source.parent),*unit['prefix'][1:]];before=B/(unit['name']+'.before.d');obj=B/(unit['name']+'.o')
   run(prefix+['-M','-MF',str(before),'-MT',str(obj),str(source)],unit['name']+'-M');names=deps(before);assert names,'empty fresh-M';snapshot={}
   for p in names:
    h=observed.get(p)
    if h is None:h=H(p);observed[p]=h
    snapshot[p]=h
    if allowed.get(p)!=h:unmatched.append(dict(TU=unit['name'],path=p,SHA256=h,expected=allowed.get(p)))
   fresh[unit['name']]=snapshot
  q.update(fresh_dependencies_SHA256=fresh,unmatched_dependencies=unmatched);save();assert not unmatched,('All changed TU dependency mismatches before any compile',unmatched);q['all_fresh_M_before_compile']=True;save()
  built={}
  for unit in plan['units']:
   source=D/unit['source'];obj=B/(unit['name']+'.o');md=B/(unit['name']+'.d');prefix=[unit['prefix'][0],'-I'+str(source.parent),*unit['prefix'][1:]];run(prefix+['-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],unit['name']+'-compile');assert deps(md)==set(fresh[unit['name']])and all(H(p)==h for p,h in fresh[unit['name']].items());built[unit['name']]=dict(object=str(obj),object_SHA256=H(obj),MD=str(md),MD_SHA256=H(md),source=str(source),source_SHA256=H(source));q['compiled']=built;q['new_'+unit['kind']]+=1;save()
  libs={}
  for item in plan['links']:
   lib=B/item['output'];lib.parent.mkdir(parents=True,exist_ok=True);args=subst(item['argv'],built,libs);args[args.index('-o')+1]=str(lib)
   objects=[a for a in args if a.endswith('.o')];assert len(objects)==item['object_count']and len(set(objects))==len(objects);assert max(args.index(p)for p in objects)<min(i for i,p in enumerate(args)if p.startswith('-l')or p in {v['path']for v in libs.values()})
   run(args,item['name']+'-link');text=run(['nm','-D','--defined-only',str(lib)],item['name']+'-symbols')[0];old=run(['nm','-D','--defined-only',item['baseline']],item['name']+'-parent-symbols')[0];now,prior=strong(text),strong(old);assert all(now.get(k)==v for k,v in prior.items()),'parent strong export lost';assert all(now.get(n)=='T'for n in item.get('required_exports',[]))
   elf=run(['readelf','-d',str(lib)],item['name']+'-ELF')[0];parentelf=run(['readelf','-d',item['baseline']],item['name']+'-parent-ELF')[0];expected_needed=needed(parentelf)
   for oldpath,newname in item.get('replace_resolution',{}).items():
    assert expected_needed.count(oldpath)==1,('unregistered Native NEEDED replacement',oldpath);expected_needed=[libs[newname]['path']if n==oldpath else n for n in expected_needed]
   assert needed(elf)==expected_needed,'NEEDED changed outside exact registered Native identity';assert 'SYMBOLIC'in elf
   env=os.environ.copy();env['LD_LIBRARY_PATH']=str(lib.parent)+':'+os.environ['LD_LIBRARY_PATH']
   if plan.get('ldd_preload'):env['LD_PRELOAD']=plan['ldd_preload'];assert H(env['LD_PRELOAD'])==plan['ldd_preload_SHA256']
   out,err=run(['/usr/bin/ldd','-r',str(lib)],item['name']+'-ldd',env);actual={p:H(p)for p in paths(out+'\n'+err)};expected=dict(plan['dynamic_SHA256'])
   for oldpath,newname in item.get('replace_resolution',{}).items():
    expected.pop(oldpath);expected[libs[newname]['path']]=libs[newname]['SHA256']
   for p in item.get('remove_resolution',[]):expected.pop(p)
   assert actual==expected,('loader exact path/bytes drift',set(actual)-set(expected),set(expected)-set(actual))
   libs[item['name']]=dict(path=str(lib),SHA256=H(lib),all_linked_object_SHA256={p:H(p)for p in objects},strong_parent_exports_retained=len(prior),NEEDED=needed(elf),all_transitive_resolved_library_SHA256=actual);q['libraries']=libs;save()
  maps=[v for v in Path('/proc/self/maps').read_text().splitlines()if re.search(r'lib(?:infiniops|infinirt|infiniccl|infinilm_runtime|torch|amdhip64|cudart|Kimi|GLM)',v)];assert not maps and 'torch'not in sys.modules;q.update(passed=True,all_fresh_M_equal_MD=True,project_maps=maps,Torch_imported=False,qualification='new changed-TU compile/link/exports/exactENV ldd only, no originalcontracts/Native/SDK/Torch/device/weights/Model/numeric/performance executed')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','new_HOST','new_GPU','elapsed_s','error']}));return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
