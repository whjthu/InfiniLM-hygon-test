"""CPU map constructor: accepted B4/I512 past0; canonical allselected policy."""
import numpy as np
def maps(kind):
 assert kind in ['swa','hca128','csa4'];B,I,sw=4,512,128;ratio={'swa':0,'hca128':128,'csa4':4}[kind];cw=1 if not ratio else 4 if ratio==128 else 512;cr=1 if not ratio else B*I//ratio
 si=np.full((B*I,sw),-1,np.int64);ci=np.full((B*I,cw),-1,np.int64);lens=np.zeros((B*I,2),np.int32)
 for request in range(B):
  for pos in range(I):
   t=request*I+pos;ns=min(pos+1,sw);nc=(pos+1)//ratio if ratio else 0;lens[t]=[ns,nc];si[t,:ns]=request*I+np.arange(pos+1-ns,pos+1)
   if nc:ci[t,:nc]=request*(I//ratio)+np.arange(nc)
 return si,ci,lens,cr,ratio
def validate():
 rows=[]
 for kind in ['swa','hca128','csa4']:
  si,ci,lens,cr,ratio=maps(kind)
  for t,(ns,nc)in enumerate(lens):
   req,pos=divmod(t,512);assert ns==min(pos+1,128)and nc==((pos+1)//ratio if ratio else 0)
   assert si[t,:ns].tolist()==list(range(req*512+pos+1-ns,req*512+pos+1))and np.all(si[t,ns:]==-1)
   assert ci[t,:nc].tolist()==(list(range(req*(512//ratio),req*(512//ratio)+nc))if ratio else[])and np.all(ci[t,nc:]==-1)
   assert len(set(si[t,:ns]))==ns and len(set(ci[t,:nc]))==nc
  # Query/request splits preserve global IDs byte-for-byte; no bank rebasing.
  for width in [32,64,512]:
   for a in [si,ci,lens]:assert np.array_equal(np.concatenate([a[i:i+width]for i in range(0,2048,width)]),a)
  rows.append({'kind':kind,'T':2048,'sw':128,'cw':ci.shape[1],'compressed_rows':cr,'ratio':ratio})
 return rows
if __name__=='__main__':print(validate())
