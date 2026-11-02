"""Actual consumer, independent constant-score page fixture and negative mutations."""
import pathlib,json,copy,importlib.util,numpy as np
S=pathlib.Path(__file__).resolve().parent
def load(n,p):sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
oracle=load('oracle390',S/'reference.py');consumer=load('consumer390',S/'consumer.py')
def field(a):return dict(shape=a.shape,raw=a.view(np.uint8).reshape(-1).copy(),dtype=-1)
def fixtures(past,previous=None):
 before,after=past//4,(past+1)//4;raw=np.full((128,8448),0xa5,np.uint8)
 for slot in range(before):
  page,row=divmod(slot,64);raw[page,row*128:(row+1)*128]=56;raw[page,8192+row*4:8196+row*4]=np.array([1],np.float32).view(np.uint8)
 state=np.zeros(64,np.uint8)
 if previous is not None:raw=consumer.array(previous[0]['post_commit'][10]).copy();state=consumer.array(previous[0]['post_commit'][11]).copy()
 tail=np.full((1,128),48,np.uint8)if after>before else None;scale=np.ones((1,1),np.float32)if tail is not None else None
 # FP8 code56=1, code48=.5; 128 dot dimensions *8 localheads *8 TPranks.
 scores=np.full((1,after),8192,np.float32)
 if past>=8575:scores[0,2143]=4096
 fields={0:field(np.full((1,8,128),56,np.uint8)),1:field(np.ones((1,8),np.float32)),2:field(tail)if tail is not None else None,3:field(scale)if scale is not None else None,4:field(scores),5:field(np.arange(512,dtype=np.int64).reshape(1,512)),6:field(raw),7:field(raw),8:field(state),9:field(state),10:field(raw),11:field(state),12:None,13:None,14:None,15:None,16:field(np.array([past],np.int64))}
 rows=[]
 for rank in range(8):
  ident=dict(generation=1 if past==8575 else 2,past=past,before=before,after=after,Runtime=100+rank,C4owner=200+rank,stream=300+rank,Pending=400+rank,frozen_raw=500+rank,query=600+rank,merged_weights=700+rank,tail_values=800+rank if tail is not None else 0,tail_scales=900+rank if tail is not None else 0,positions=1000+rank,active=True,committed=False,original_commit_validates_private_owner_revision=True)
  a=copy.deepcopy(fields);b=copy.deepcopy(fields);c=copy.deepcopy(fields)
  b[12]=field(scores);b[13]=field(np.arange(512,dtype=np.int64).reshape(1,512));c=copy.deepcopy(b)
  c[14]=field(scores);c[15]=field(np.arange(512,1024,dtype=np.int64).reshape(1,512)) # Different valid tiedTopK accepted.
  post=copy.deepcopy(c);newraw=raw.copy()
  if tail is not None:
   page,row=divmod(before,64);newraw[page,row*128:(row+1)*128]=tail[0];newraw[page,8192+row*4:8196+row*4]=scale.reshape(-1).view(np.uint8)
  post[10]=field(newraw);post[11]=field(np.full(64,1 if past==8575 else 2,np.uint8))
  rows.append(dict(rank=rank,identity_before=ident,identity_after_sdk=copy.deepcopy(ident),identity_after_paged=copy.deepcopy(ident),identity_post_commit=dict(ident,active=False,committed=True),before=a,after_sdk=b,after_paged=c,post_commit=post))
 return rows
checks=[];first=fixtures(8575);second=fixtures(8576,first)
for name,rows,previous in [('newtail_boundary',first,None),('nexttoken_consumes_committed_cache_state',second,first)]:
 result=consumer.validate_same_cache(rows,oracle,previous);assert result['passed'];checks.append(name)
for kind in ['field6_changed','query_changed','lease_identity_changed','duplicate_TopK','nonfinite_score','wrong_commit_tail','next_cache_not_carried','next_state_not_carried']:
 rows=copy.deepcopy(second if kind.startswith('next_')else first);previous=first if kind.startswith('next_')else None
 if kind=='field6_changed':rows[0]['after_paged'][6]['raw'][0]^=1
 if kind=='query_changed':rows[0]['after_paged'][0]['raw'][0]^=1
 if kind=='lease_identity_changed':rows[0]['identity_after_paged']['frozen_raw']+=1
 if kind=='duplicate_TopK':rows[0]['after_paged'][15]['raw'].view(np.int64)[1]=rows[0]['after_paged'][15]['raw'].view(np.int64)[0]
 if kind=='nonfinite_score':rows[0]['after_paged'][14]['raw'].view(np.float32)[0]=float('nan')
 if kind=='wrong_commit_tail':rows[0]['post_commit'][10]['raw'][0]^=1
 if kind=='next_cache_not_carried':
  for stage in ['before','after_sdk','after_paged']:
   for key in [6,7,10]:rows[0][stage][key]['raw'][0]^=1
  rows[0]['post_commit'][10]['raw'][0]^=1
 if kind=='next_state_not_carried':
  for stage in ['before','after_sdk','after_paged']:
   for key in [8,9,11]:rows[0][stage][key]['raw'][0]^=1
 try:consumer.validate_same_cache(rows,oracle,previous)
 except(AssertionError,ValueError):checks.append('reject_'+kind)
 else:raise AssertionError('Accepted '+kind)
padding=copy.deepcopy(first);padding[0]['after_paged'][6]['raw'][-1]^=1;assert consumer.validate_same_cache(padding,oracle)['passed'];checks.append('unused_raw_padding_not_false_numeric_failure')
pre=copy.deepcopy(first)
for r in pre:r.pop('post_commit');r.pop('identity_post_commit')
assert consumer.validate_same_cache(pre,oracle,committed=False)['passed'];checks.append('both_scores_checked_before_commit')
print(json.dumps(dict(completed=True,passed=True,checks=checks,actual_consumer=True,constant_FP8_score_independent_expected=True,field6_valid_prefix_frozen=True,TopK_ties_different_sets_accepted=True,GPU=False,Native_dlopen=False,actual_GPU_numeric='NOT_RUN')))
