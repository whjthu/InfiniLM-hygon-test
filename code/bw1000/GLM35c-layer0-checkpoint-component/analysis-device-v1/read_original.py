"""Read one sealed genuine-Layer0 archive; CPU stdlib only, no device runtime."""
from pathlib import Path
import argparse,array,hashlib,heapq,json,math,struct,sys,tarfile
sha=lambda b:hashlib.sha256(b).hexdigest()
def fp(raw,dtype):
 if dtype==0:
  a=array.array('f');a.frombytes(raw)
  if sys.byteorder!='little':a.byteswap()
  return a
 assert dtype==1
 a=array.array('H');a.frombytes(raw)
 if sys.byteorder!='little':a.byteswap()
 return [struct.unpack('<f',struct.pack('<I',x<<16))[0]for x in a]
def ids(raw):return list(struct.unpack('<'+'q'*(len(raw)//8),raw))
def delta(x,y,dtype):
 a,b=fp(x,dtype),fp(y,dtype);assert len(a)==len(b)and all(math.isfinite(t)for t in a)and all(math.isfinite(t)for t in b)
 n=len(a);d=math.fsum((u-v)**2 for u,v in zip(a,b));base=math.fsum(u*u for u in a)
 return dict(elements=n,maxabs=max((abs(u-v)for u,v in zip(a,b)),default=0.0),NRMS=math.sqrt(d)/max(math.sqrt(base),1e-30),NRMS_denominator='max(L2(serial),1e-30)',serial_L2=math.sqrt(base),dtype='F32'if dtype==0 else'BF16',same_bits=x==y,threshold_applied=False)
def analyze(path):
 with tarfile.open(path,'r:gz')as tar:
  members={m.name:m for m in tar.getmembers()};assert all(m.isfile()and not Path(n).is_absolute()and'..'not in Path(n).parts for n,m in members.items())
  raw=lambda n:tar.extractfile(members[n]).read()
  j=lambda n:json.loads(raw(n))
  seal=j('SEAL.json');assert seal['sealed_terminal']and seal['GPU_label']=='GLM35c-layer0-checkpoint-device-v1'
  for name,r in seal['files'].items():
   b=raw(name);assert len(b)==r['bytes']and sha(b)==r['SHA256'],name
  w=j('watch.json');acct=j('accounting.json');entry=j('entry.json')if'entry.json'in members else{};source=j('GPU_SOURCE_MANIFEST.json');cpu=j('CPU-result.json')
  assert sha(raw('GPU_SOURCE_MANIFEST.json'))=='40ccf80c533a49a6be3a3d399c9d0a1f7c53d6376551edbb58e863cffeaab4ae'
  assert sha(raw('CPU-result.json'))=='00e8f65a5a5b8ed650854e2fc8b1e3fb7354bdfc7d68b8348286e513e06aa8fe'and cpu['library_SHA256']==source['library_SHA256']
  out=dict(archive_SHA256=sha(Path(path).read_bytes()),sealed_files=len(seal['files']),original_watch_passed=w.get('passed'),original_completed=w.get('completed'),actual_step=w.get('actual_step'),actual_GPU_claim=source['resource'],component_only=True,Model_quality=False,performance=False,rootcause='UNKNOWN; isolated Layer0 bit/numeric observations do not identify fullModel cause',operational_and_lifecycle_passed=False,owners=[])
  if not w.get('passed'):
   out.update(error=w.get('error'),forced_cleanup=w.get('forced_cleanup'),partial=True);return out
  assert w['completed']and w['normal8']and not w.get('forced_cleanup',False)and w['srun_exit0']and w['Slurm_COMPLETED0']
  assert entry['normal']and entry['normal8']and entry['all_owner_threads_processes_joined']
  assert len(acct)==len({r[1]for r in acct})
  rows={r[1]:r for r in acct};label=seal['GPU_label']
  for suffix in ['', '-keeper','-post']:
   r=rows[label+suffix];assert r[2:4]==['COMPLETED','0:0']
  assert rows[label][0]==w['actual_step']and rows[label][5]=='a02r3n03'and rows[label+'-keeper'][5]=='a14r2n20'
  post=w['post'];assert post['passed']and post['KFD_available']is True and not post['KFD_pids']and not post['matching_GPU_driver_PIDs']and not post['expected_PIDs_still_exist']and all(post['ports_bindable'].values())and post['actual_step']==rows[label+'-post'][0]and rows[label+'-post'][5]=='a02r3n03'
  # Decode the original published freezer, never a provider-specific reconstruction.
  scope={'__name__':'pinned_original_freezer'};exec(compile(raw('freeze_continuation_inputs.py'),'pinned_original_freezer','exec'),scope)
  frozen=scope['freeze'](raw('original-private-plan.json'));assert sha(frozen)==source['frozen_plan_SHA256']and sha(raw('original-private-plan.json'))==source['original_plan_SHA256']
  above=next(x for x in json.loads(frozen)['cases']if x['geometry']=='above');assert len(above['prompt'])==2949
  trace=[]
  for width in [1,8]:
   trace.append(('reset',));past=0
   for n in above['chunk_widths']:
    a=above['prompt'][past:past+n];trace.append(('step',past,n,sha(struct.pack('<'+'q'*n,*a)),0));past+=n
   a=above['proposal'][:width];trace.append(('step',past,width,sha(struct.pack('<'+'q'*width,*a)),1))
  trace.append(('reset',));total=0;cpus=[];pcis=[]
  for rank in range(8):
   a=j('rank%d.json'%rank);assert a['completed']and a['normal']and a['normal_destroy']and a['Runtime_weak_expired']and a['Torch_after_destroy']and a['owner_device_contract']and a['owner_TID']==a['PID']and a['SDK_first_submit_affinity_observed']
   assert a['rank']==rank and a['actual_step']==w['actual_step']and a['GPU_source_manifest_SHA256']==sha(raw('GPU_SOURCE_MANIFEST.json'))and a['library_SHA256']==source['library_SHA256']
   common=entry['pre_bootstrap_common_CPU_affinity'];assert len(common)>=32 and a['pre_first_submit_common_CPU_affinity']==common and a['owner_CPU_affinity']==common[rank*4:rank*4+4];cpus+=a['owner_CPU_affinity'];pcis.append(a['physical_PCI'])
   assert a['gcnArchName'].split(':')[0]=='gfx936'and a['free_after_load'][0]>4*1024**3
   assert a['actual_qualified_dependencies']==source['qualified_dependency_SHA256']
   assert set(a['actual_symbol_owners'])=={'glm_l0_'+n for n in ['error','id_size','id','create','keys','load','finish','step','snapshot','read','reset','destroy']}and set(a['actual_symbol_owners'].values())=={cpu['library']}
   assert len(a['load']['checkpoint_keys'])==21 and a['actual_prompt_tokens']==2949 and a['actual_prefix_chunk_widths']==above['chunk_widths']
   actual=[]
   for t in a['Native_trace']:
    assert t['RC']==0
    actual.append(('reset',)if t['kind']=='reset'else('step',t['past'],t['tokens'],t['input_i64_SHA256'],t['capture']))
   assert actual==trace
   snaps={}
   for snap in a['snapshots']:
    width=snap['proof']['width'];assert width in [1,8]and width not in snaps
    f={}
    for r in snap['files']:
     name='raw/'+Path(r['path']).name;b=raw(name);assert sha(b)==r['SHA256']and len(b)==r['bytes'];total+=len(b)
     assert r['bytes']==math.prod(r['shape'])*({0:4,1:2,3:8}[r['dtype']]);f[r['name']]=dict(r,raw=b)
     if r['dtype']!=3:
      if r['dtype']==1:
       bits=array.array('H');bits.frombytes(b)
       if sys.byteorder!='little':bits.byteswap()
       assert all((x&0x7f80)!=0x7f80 for x in bits),(rank,width,r['name'],'nonfinite BF16 exponent')
      else:
       values=fp(b,0)
       for index,v in enumerate(values):
        future=r['name']=='selector.masked_F32_scores'and index%(2949+width)>2949+index//(2949+width)
        assert v==-math.inf if future else math.isfinite(v),(rank,width,r['name'],index)
    assert len(f)==21
    selected=ids(f['selection.IDs']['raw']);score=fp(f['selector.masked_F32_scores']['raw'],0)
    for row in range(width):
     ss=selected[row*2048:(row+1)*2048];assert len(set(ss))==2048 and all(0<=k<=2949+row for k in ss)
     vv=score[row*(2949+width):(row+1)*(2949+width)];ssset=set(ss)
     assert min(vv[k]for k in ss)>=max(vv[k]for k in range(2950+row)if k not in ssset)
    union=ids(f['NoV.union_ids']['raw']);assert union==sorted(set(selected))and f['NoV.union_V']['shape']==[len(union),8,256]
    snaps[width]=f
   assert set(snaps)=={1,8}
   x,y=snaps[1],snaps[8];r=dict(rank=rank,history={},query0={},selection={},first_query0_bit_difference=None)
   for name in ['history.latent','history.rotary','history.indexer','history.projected_K']:r['history'][name]=x[name]['SHA256']==y[name]['SHA256']
   order=['input.embedding','input.normalized_attention','prepared.q_resid','prepared.Q','prepared.latent','prepared.rotary','indexer.query','indexer.key','indexer.head_weights','selector.masked_F32_scores','selection.IDs','fresh.K','fresh.V']
   sa,sb=ids(x['selection.IDs']['raw']),ids(y['selection.IDs']['raw'])[:2048];common_ids=sorted(set(sa)&set(sb));r['selection']=dict(same_order=sa==sb,same_set=set(sa)==set(sb),intersection=len(common_ids),union_U_serial=len(ids(x['NoV.union_ids']['raw'])),union_U_width8=len(ids(y['NoV.union_ids']['raw'])))
   for name in order+['attention.NoV_output','attention.Oproj_TP_output']:
    n=x[name]['bytes'];u,v=x[name]['raw'],y[name]['raw'][:n]
    r['query0'][name]=dict(same_bits=u==v,compared_bytes=n)if x[name]['dtype']==3 else delta(u,v,x[name]['dtype'])
   def canonical(f):
    lookup={k:i for i,k in enumerate(ids(f['NoV.union_ids']['raw']))};v=f['NoV.union_V']['raw'];return b''.join(v[lookup[k]*4096:(lookup[k]+1)*4096]for k in common_ids)
   r['query0']['NoV.canonical_common_query0_V']=delta(canonical(x),canonical(y),1)
   order+=['NoV.canonical_common_query0_V','attention.NoV_output','attention.Oproj_TP_output'];r['first_query0_bit_difference']=next((n for n in order if not r['query0'][n]['same_bits']),None);r['bit_difference_not_rootcause_or_quality_threshold']=True;out['owners'].append(r)
  assert len(set(cpus))==32 and len(set(pcis))==8 and set(pcis)==set(w['before']['physical_PCI'])and total<=512*1024**2
  out.update(operational_and_lifecycle_passed=True,normal_owners=8,raw_bytes=total,all_fields_finite_except_causal_mask=True,actual_original_reset_and_ID_trace_verified=True,history_scope='Postforward retained oldvalidprefix views only; no preforward immutable clone',earliest_bits_by_rank=[x['first_query0_bit_difference']for x in out['owners']]);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=analyze(a.archive);Path(a.output).open('x',encoding='utf8').write(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q[k]for k in ['operational_and_lifecycle_passed','actual_step','Model_quality','performance','rootcause']}))
