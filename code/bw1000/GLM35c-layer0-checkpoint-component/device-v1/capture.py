"""CPU-only interpretation of actual completed component bytes, no GPU math."""
import ctypes as C,hashlib,json,math,struct,time
from pathlib import Path
P=2949;TOPK=2048
NAMES={'history.latent','history.rotary','history.indexer','history.projected_K','input.embedding','input.normalized_attention','indexer.query','indexer.key','indexer.head_weights','selector.masked_F32_scores','fresh.K','fresh.V','NoV.union_ids','NoV.union_V','prepared.Q','prepared.q_resid','prepared.latent','prepared.rotary','selection.IDs','attention.NoV_output','attention.Oproj_TP_output'}
def sha(b):return hashlib.sha256(b).hexdigest()
def integers(b):return [a[0]for a in struct.iter_unpack('<q',b)]
def floats(b):return [a[0]for a in struct.iter_unpack('<f',b)]
def finite_field(name,row,b,width):
 assert len(b)==row['bytes']and len(b)==math.prod(row['shape'])*{0:4,1:2,3:8}[row['dtype']]
 if row['dtype']==1:assert all((x[0]&0x7f80)!=0x7f80 for x in struct.iter_unpack('<H',b)),name+' BF16nonfinite'
 if row['dtype']==0:
  values=floats(b)
  if name=='selector.masked_F32_scores':
   assert row['shape']==[width,P+width]
   for i in range(width):
    for j in range(P+width):
     v=values[i*(P+width)+j]
     assert (math.isfinite(v)if j<=P+i else v==-math.inf),'actual score valid/future mask differs'
  else:assert all(math.isfinite(v)for v in values),name+' F32nonfinite'
def validate_snapshot(fields,width):
 assert set(fields)==NAMES
 expected={'history.latent':[P,512],'history.rotary':[P,64],'history.indexer':[P,128],'history.projected_K':[P,8,256],'input.embedding':[width,6144],'input.normalized_attention':[width,6144],'indexer.query':[width,4,128],'indexer.key':[width,128],'indexer.head_weights':[width,4],'selector.masked_F32_scores':[width,P+width],'fresh.K':[width,8,256],'fresh.V':[width,8,256],'prepared.Q':[width,8,256],'prepared.q_resid':[width,2048],'prepared.latent':[width,512],'prepared.rotary':[width,1,64],'selection.IDs':[width,TOPK],'attention.NoV_output':[width,8,256],'attention.Oproj_TP_output':[width,6144]}
 for n,f in fields.items():
  assert f['dtype']==(0 if n in ['indexer.head_weights','selector.masked_F32_scores']else 3 if n in ['selection.IDs','NoV.union_ids']else 1),n
  if n in expected:assert f['shape']==expected[n],n
 shape=lambda n:fields[n]['shape'];raw=lambda n:fields[n]['raw']
 assert shape('selection.IDs')==[width,TOPK]
 ids=integers(raw('selection.IDs'));scores=floats(raw('selector.masked_F32_scores'));margins=[]
 for i in range(width):
  selected=ids[i*TOPK:(i+1)*TOPK];assert len(set(selected))==TOPK and all(0<=k<=P+i for k in selected)
  values=scores[i*(P+width):(i+1)*(P+width)];chosen=set(selected);lo=min(values[k]for k in chosen);hi=max(values[k]for k in range(P+i+1)if k not in chosen)
  assert lo>=hi,'actual TopK set inconsistent with its F32 scores';margins.append(lo-hi)
 union=integers(raw('NoV.union_ids'));assert union==sorted(set(ids))and len(union)<=P+width
 assert shape('NoV.union_V')==[len(union),8,256]
 return dict(width=width,actual_U=len(union),TopK=TOPK,valid_F32_selector_margin=margins,TopK_checked_on_observed_F32_scores=True,mask_future_negative_infinity=True,all_other_float_fields_finite=True)
class Trace:
 def __init__(self,lib,above,root,rank,deadline):
  self.lib,self.root,self.rank,self.deadline=lib,Path(root),rank,deadline;self.calls=[];self.snapshots=[];self.fields={};self.used=0;self.expected=[]
  for width in [1,8]:
   past=0
   for n in above['chunk_widths']:
    values=above['prompt'][past:past+n];self.expected.append((past,n,sha(struct.pack('<'+'q'*n,*values)),0));past+=n
   values=above['proposal'][:width];self.expected.append((past,width,sha(struct.pack('<'+'q'*width,*values)),1))
 def __getattr__(self,n):return getattr(self.lib,n)
 def limit(self):
  if time.time()>=self.deadline:raise TimeoutError('original owner540s epoch exhausted')
 def glm_l0_step(self,ctx,past,n,inputs,capture):
  self.limit();i=sum(x['kind']=='step'for x in self.calls);b=C.string_at(inputs,n*8)
  assert i<len(self.expected)and(past,n,sha(b),capture)==self.expected[i],'original input/past/capture trace differs'
  started=time.time();rc=self.lib.glm_l0_step(ctx,past,n,inputs,capture);self.calls.append(dict(kind='step',past=past,tokens=n,input_i64_SHA256=sha(b),capture=capture,RC=int(rc),started_epoch=started,returned_epoch=time.time()));return rc
 def glm_l0_reset(self,ctx):
  self.limit();rc=self.lib.glm_l0_reset(ctx);self.calls.append(dict(kind='reset',RC=int(rc),epoch=time.time()));return rc
 def glm_l0_snapshot(self,ctx):
  self.limit();raw=self.lib.glm_l0_snapshot(ctx)
  if raw is None:return raw
  self.meta=json.loads(raw);assert self.meta['component_only']is True and[self.meta[k]for k in ['layer','TP','PP','world']]==[0,8,1,8]
  self.width=self.meta['past_after']-P;assert self.width in [1,8];rows=self.meta['fields'];assert len(rows)==21 and{r['name']for r in rows}==NAMES
  self.fields={};return raw
 def glm_l0_read(self,ctx,index,out,n):
  self.limit();rc=self.lib.glm_l0_read(ctx,index,out,n)
  if rc:return rc
  row=self.meta['fields'][index];b=C.string_at(out,n);finite_field(row['name'],row,b,self.width);self.fields[row['name']]=dict(row,raw=b,SHA256=sha(b))
  if index==len(self.meta['fields'])-1:
   proof=validate_snapshot(self.fields,self.width);files=[]
   for i,row in enumerate(self.meta['fields']):
    f=self.fields[row['name']];self.used+=len(f['raw']);assert self.used<=64*1024**2,'perowner two-snapshot64MiB raw cap'
    path=self.root/('rank%d-width%d-field%02d.bin'%(self.rank,self.width,i))
    with path.open('xb')as z:z.write(f['raw'])
    files.append(dict(name=f['name'],dtype=f['dtype'],shape=f['shape'],bytes=len(f['raw']),SHA256=f['SHA256'],path=str(path)))
   self.snapshots.append(dict(metadata=self.meta,proof=proof,files=files,raw_total_so_far=self.used))
  return rc
def compare(actual):
 a,b=actual[1]['fields'],actual[8]['fields'];eq={}
 history=['history.latent','history.rotary','history.indexer','history.projected_K']
 for n in history:eq[n]=a[n]['SHA256']==b[n]['SHA256']
 stages=['input.embedding','input.normalized_attention','prepared.q_resid','prepared.Q','prepared.latent','prepared.rotary','indexer.query','indexer.key','indexer.head_weights','selector.masked_F32_scores','selection.IDs','fresh.K','fresh.V','attention.NoV_output','attention.Oproj_TP_output']
 query={}
 for n in stages:
  count=(P+1)*4 if n=='selector.masked_F32_scores'else a[n]['bytes'];x=a[n]['raw'][:count];y=b[n]['raw'][:count];query[n]=dict(same_bits=x==y,serial_SHA256=sha(x),width8_query0_SHA256=sha(y),compared_bytes=count)
 sa=integers(a['selection.IDs']['raw'])[:TOPK];sb=integers(b['selection.IDs']['raw'])[:TOPK];common=sorted(set(sa)&set(sb))
 def canonical(fields):
  ids=integers(fields['NoV.union_ids']['raw']);lookup={k:i for i,k in enumerate(ids)};v=fields['NoV.union_V']['raw'];return b''.join(v[lookup[k]*4096:(lookup[k]+1)*4096]for k in common)
 av,bv=canonical(a),canonical(b);query['NoV.canonical_common_query0_V']=dict(same_bits=av==bv,serial_SHA256=sha(av),width8_SHA256=sha(bv),common_selected_IDs=len(common),compared_bytes=len(av))
 order=stages[:stages.index('attention.NoV_output')]+['NoV.canonical_common_query0_V']+stages[stages.index('attention.NoV_output'):]
 earliest=next((n for n in order if not query[n]['same_bits']),None)
 return dict(postforward_valid_old_prefix_equal=all(eq.values()),history_by_field=eq,query0_by_stage=query,selection_same_order=sa==sb,selection_same_set=set(sa)==set(sb),earliest_observed_query0_bit_difference=earliest,bit_difference_is_not_a_failure_threshold_or_rootcause=True,NoV_U_serial=len(integers(a['NoV.union_ids']['raw'])),NoV_U_width8=len(integers(b['NoV.union_ids']['raw'])),diagnostic_only=True,Model_quality=False,performance=False,rootcause='UNKNOWN; isolated component observation only',history_scope='Retained oldvalidprefix views copied after original forward drain, not pre-forward immutable clones')
