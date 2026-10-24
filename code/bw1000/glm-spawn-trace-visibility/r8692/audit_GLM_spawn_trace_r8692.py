import collections,hashlib,json,pathlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';state=r/'results/GLM-spawn-trace-r8692-state.json';s=json.loads(state.read_text())
q={'GPU_performance_claim':False,'no_Model_no_weights_no_CCL':True,'same_original_entry_only':True,'state':s,'trace_visible_all8_children':False}
if s.get('terminal_observed')and len(s.get('rows',[]))==1:
 tag=s['rows'][0][0];b=r/'results'/f'GLM-spawn-trace-r8692-{tag}';q['actual_handle']=tag
 files=sorted(b.rglob('*'))if b.exists()else[];q['raw_bytes']=sum(p.stat().st_size for p in files if p.is_file());assert q['raw_bytes']<=256*1024**2
 q['file_receipts']={str(p.relative_to(r)):{'bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files if p.is_file()}
 owners=[]
 for rank in range(8):
  p=r/'results'/f'GLM-spawn-trace-r8692-{tag}-rank{rank}.json'
  if p.exists():owners.append(json.loads(p.read_text()))
 q['owners']=owners;kernel=collections.Counter();samples={}
 for p in files:
  if p.suffix!='.json':continue
  obj=json.loads(p.read_text());events=obj.get('traceEvents',[])if isinstance(obj,dict)else[]
  for e in events:
   if e.get('ph')=='X'and e.get('cat')=='HIPOPS':
    a=e.get('args',{});key=(int(a['pid']),int(a['devId']));kernel[key]+=1;samples.setdefault(str(key),e)
 q['kernel_counts']=[{'pid':pid,'device':dev,'count':count}for(pid,dev),count in kernel.items()];q['kernel_samples']=samples
 q['trace_visible_all8_children']=bool(len(owners)==8 and len({o['pid']for o in owners})==8 and all(o['completed']and o['start_code']==o['stop_code']==0 and o['sum']==120 and kernel[o['pid'],o['rank']]>0 for o in owners))
 q['normal8children_returned']=bool(len(owners)==8 and s['numeric']['exit']==0 and s['rows'][0][2:4]==['COMPLETED','0:0'])
 q['original_post_empty']=bool(s.get('original_post',{}).get('exit')==0 and any(json.loads(v).get('kfd_pids')==[]for v in s['original_post']['stdout'].splitlines()if v.startswith('{')))
 q['qualified_tool_visibility']=q['trace_visible_all8_children']and q['normal8children_returned']and q['original_post_empty']
print(json.dumps(q))
