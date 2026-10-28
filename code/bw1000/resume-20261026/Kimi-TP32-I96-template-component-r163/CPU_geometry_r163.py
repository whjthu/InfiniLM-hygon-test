import json
from pathlib import Path
E,H,I,T,TP=896,3584,96,64,32
align=lambda n:(n+255)//256*256
counts=align(T*16*I*2);routes=align(counts+E*4);expert=align(routes+E*T*16*4);start=align(expert+T*16*4);jobs=align(start+T*16*4);total=align(jobs+4)
assert I*TP==3072 and H//2==1792 and H//32==112 and I//2==48 and I//32==3
assert total<4*1024**2 and len({r*48+p for r in range(TP)for p in range(48)})==1536
assert len({r*3+p for r in range(TP)for p in range(3)})==96
for e in [0,895]:
 for h in [0,3583]:
  for p in [0,15,16,31,32,47]:
   assert (e*H+h)*48+p<E*H*48 and (e*H+h)*3+p//16<E*H*3
assert 2*I//4==48 and (T*16+1)//2<=T*16
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'Native_loaded':False,'TP':TP,'local_I':I,'workspace_bytes':total,'offsets':dict(counts=counts,routes=routes,expert=expert,start=start,jobs=jobs),'scope':'geometry and finite workspace only; not GPU or Modelfit'}))
