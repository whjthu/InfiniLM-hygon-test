def microbatches(active,M):
 assert 1<=M<=32 and len(active)==len(set(active)) and all(0<=i<32 for i in active)
 return [active[i:i+M]for i in range(0,len(active),M)]

def prefix_chunks(prompts,rows,micro=8,limit=4096):
 assert len(prompts)==len(rows) and len(set(rows))==len(rows) and 1<=micro<=32
 for b in range(0,len(rows),micro):
  indices=list(range(b,min(b+micro,len(rows))));offset={i:0 for i in indices}
  while any(offset[i]<len(prompts[i])for i in indices):
   selected=[];tokens=[];pasts=[];budget=limit
   for i in indices:
    n=min(len(prompts[i])-offset[i],budget)
    if n:selected.append(rows[i]);tokens.append(prompts[i][offset[i]:offset[i]+n]);pasts.append(offset[i]);offset[i]+=n;budget-=n
   assert selected and sum(map(len,tokens))<=limit
   yield selected,tokens,pasts

def selftest():
 rows=[0,3,16,31];prompts=[list(range(n))for n in [1,511,638,200]];seen={i:[]for i in rows}
 for rr,tt,pp in prefix_chunks(prompts,rows):
  for i,t,p in zip(rr,tt,pp):assert p==len(seen[i]);seen[i]+=t
 assert all(seen[i]==v for i,v in zip(rows,prompts));assert microbatches(list(range(32)),16)==[list(range(16)),list(range(16,32))];assert microbatches([0,5,17,31],8)==[[0,5,17,31]]
 return {'passed':True,'CPU_only':True,'scope':'stable logicalrows/noquestiontruncation/chunk<=4096/M16M32/C32 planning'}
