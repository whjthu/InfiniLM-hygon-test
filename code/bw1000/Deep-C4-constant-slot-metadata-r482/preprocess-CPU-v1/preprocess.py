"""One finite original50 -M dependency closure. Never compiles or links.
Original sources/include origins stay in place; no candidate header is injected.
All50 results/mismatches are collected before deciding graph qualification.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,os,re,shlex,shutil,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent
LABEL='Deep-C4-constant-slots-r482-all50-preprocess-CPU-v1'
B=Path('/refactor/candidates')/LABEL
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
HEADER_SHA='b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6'

def dep_paths(text,target):
 lines=text.replace('\\\r\n','').replace('\\\n','').splitlines();names=set();rules=0
 for line in lines:
  if not line.strip():continue
  left,colon,right=line.partition(':');assert colon and shlex.split(left)==[target],('unexpected make rule',left)
  rules+=1;names.update(shlex.split(right))
 assert rules and names and all(p.startswith('/')for p in names),'empty/nonabsolute dependency path'
 return names

def main():
 start=time.monotonic();assert not B.exists(),'Existing candidate: reconcile original, never replay';B.mkdir()
 q=dict(completed=False,passed=False,CPU_only=True,preprocess_only=True,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],Native_dlopen=False,Torch_imported=False,SDK_explicit_calls=0,Runtime_created=False,weights_opened=False,Model_qualified=False,GPU_job=False,compile_commands=0,link_commands=0,planned_TUs=50,completed_TUs=0,preprocess_commands=[],units={},failures=[],unmatched_dependencies=[],changed_historical_dependencies=[],source_mismatches=[],snapshot_SHA256={},class_header_consumers=[],class_header_variants=[],Native_CPU_GO=False,GPU_GO=False,driver_child_implicit_initialization='Not probed; compiler drivers receive -M only and all GPUvisibility=-1')
 def save():
  p=B/'CPU-result.tmp';p.write_text(json.dumps(q,indent=2)+'\n');os.replace(p,B/'CPU-result.json')
 def remaining():return 265-(time.monotonic()-start)
 try:
  claim=json.loads((D/'CPU_RESOURCE_CLAIM.json').read_bytes());assert H(D/'CPU_RESOURCE_CLAIM.json')==os.environ['TUNING_CPU_CLAIM_SHA']and claim['CPU_GO']and not claim['Native_compile_GO']and not claim['GPU_GO']and claim['preprocess_TUs']==50 and claim['parent']==os.environ['SLURM_JOB_ID']
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert H(D/'SOURCE_MANIFEST.json')==claim['source_manifest_SHA256']and all(H(D/p)==h for p,h in mf['files_SHA256'].items());q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),claim_SHA256=H(D/'CPU_RESOURCE_CLAIM.json'))
  tools=json.loads((D/'TOOLCHAIN.json').read_bytes());seen={}
  for p,pin in tools['tools'].items():
   canonical=str(Path(p).resolve());assert canonical==pin['canonical'],('tool path',p)
   if canonical not in seen:seen[canonical]=H(p)
   assert seen[canonical]==pin['sha256'],('tool bytes',p)
  for k,v in tools['required_environment'].items():assert os.environ.get(k)==v,(k,'environment drift')
  recipes=json.loads((D/'ORIGINAL_TU_RECIPES.json').read_bytes());hist=json.loads((D/'HISTORICAL_DEPENDENCIES.json').read_bytes());records=recipes['objects'];assert len(records)==50 and recipes['original_parent_actual_CPU']=='910407.750';q['original_CPU377_receipt_SHA256']=recipes['original_CPU377_receipt_SHA256']
  expected_maps={o:dict(hist['entries'][i]for i in ids)for o,ids in hist['entry_indices_by_linked_object'].items()};q['historical_maps_present']=hist['units_with_historical_maps'];tasks=[]
  for i,(obj,r)in enumerate(sorted(records.items())):
   assert str(Path(r['source']).parent)==r['quote_include_origin']
   current=H(r['source'])
   if current!=r['source_SHA256']:q['source_mismatches'].append(dict(TU=i,source=r['source'],expected=r['source_SHA256'],actual=current))
   prefix=r['preprocess_prefix'];compiler=shutil.which(prefix[0]);assert compiler==tools['compiler_selection'][prefix[0]],('compiler selection',prefix[0],compiler)
   assert '-c'not in prefix and '-o'not in prefix and '-MD'not in prefix and '-M'not in prefix and '-shared'not in prefix
   assert not any(x.endswith(('.cpp','.cu','.hip','.o'))and not x.startswith('-')for x in prefix),'source/output in prefix'
   target='TU%02d'%i;md=B/(target+'.M');args=prefix+['-M','-MF',str(md),'-MT',target,r['source']];task=dict(index=i,target=target,linked_object=obj,original_object_SHA256=r['object_SHA256'],source=r['source'],source_SHA256=current,quote_include_origin=r['quote_include_origin'],argv=args,MD=str(md));tasks.append(task);q['preprocess_commands'].append(args)
  save();assert not q['source_mismatches'],('original source changed; no preprocess was started',q['source_mismatches'])
  def run(task):
   begin=time.monotonic();left=remaining();assert left>2,'Original265s work cap'
   try:
    z=subprocess.run(task['argv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=max(.1,min(50,left-1)))
    for key,text in [('stdout',z.stdout),('stderr',z.stderr)]:
     p=B/(task['target']+'.'+key);p.write_text(text);task[key]=dict(path=str(p),bytes=p.stat().st_size,SHA256=H(p))
    task.update(rc=z.returncode,elapsed_s=time.monotonic()-begin)
    if z.returncode==0:
     md=Path(task['MD']);task.update(MD_SHA256=H(md),dependency_paths=sorted(dep_paths(md.read_text(),task['target'])))
    else:task['error_tail']=z.stderr[-1200:]
   except BaseException as e:task.update(error=repr(e),elapsed_s=time.monotonic()-begin)
   return task
  with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
   futures=[pool.submit(run,t)for t in tasks]
   for future in concurrent.futures.as_completed(futures):
    t=future.result();q['units'][t['target']]=t;q['completed_TUs']+=1
    if t.get('rc')!=0 or'error'in t:q['failures'].append(dict(TU=t['target'],rc=t.get('rc'),error=t.get('error'),tail=t.get('error_tail')))
    save()
  # Hash each observed raw path once; avoid canonical/NFS tree walks and preserve
  # aliases explicitly. A new path is recorded, never directory-whitelisted.
  consumers=[];variants=[]
  for key,t in sorted(q['units'].items()):
   if t.get('rc')!=0 or'error'in t:continue
   known=expected_maps[t['linked_object']];snapshot={};class_headers=[]
   for p in t['dependency_paths']:
    if p not in q['snapshot_SHA256']:
     assert remaining()>1,'Original265s work cap while hashing'
     q['snapshot_SHA256'][p]=H(p)
    value=q['snapshot_SHA256'][p];snapshot[p]=value;expected=known.get(p)
    # Every original source has its independent exact source pin, including the
    # fourteen units without old full MD/hash maps.
    if p==t['source']:expected=t['source_SHA256']
    row=dict(TU=key,linked_object=t['linked_object'],path=p,SHA256=value,expected=expected)
    if expected is None:q['unmatched_dependencies'].append(row)
    elif expected!=value:q['changed_historical_dependencies'].append(row)
    if value==HEADER_SHA:class_headers.append(dict(path=p,SHA256=value))
    elif Path(p).name=='deepseek_v4_c4_indexer.hpp':variants.append(row)
   t['fresh_dependencies_SHA256']=snapshot
   assert t['source']in snapshot and snapshot[t['source']]==t['source_SHA256']
   if class_headers:consumers.append(dict(TU=key,linked_object=t['linked_object'],original_object_SHA256=t['original_object_SHA256'],source=t['source'],quote_include_origin=t['quote_include_origin'],class_headers=class_headers))
  q.update(class_header_consumers=consumers,class_header_variants=variants,all50_fresh_M_complete=q['completed_TUs']==50 and not q['failures'],known_historical_pins_match=not q['changed_historical_dependencies'],all_current_dependency_hashes_recorded=not q['failures'],class_size_compatible=False)
  assert not list(B.glob('*.o'))and not list(B.glob('*.so')),'No object/library output permitted'
  maps=[s for s in Path('/proc/self/maps').read_text().splitlines()if re.search(r'lib(?:infiniops|infiniccl|infinirt|infinilm_runtime|torch|cudart|amdhip64)',s)];assert not maps and 'torch'not in sys.modules;q['parent_project_maps']=maps
  q['passed']=q['all50_fresh_M_complete']and q['known_historical_pins_match']and not q['class_header_variants']
  q['qualification']='Original50 preprocessing graph/current SHA snapshot only. Unmatched historical paths are preserved for explicit review; this does not authorize any Native compile, binary reuse, class-layout qualification, SDK/device/numeric/Model execution.'
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start);save()
 print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','completed_TUs','compile_commands','elapsed_s','error']}));return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
