import pathlib,json
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/candidates/GLM-operator-search-r341/fused.so';b=p.read_bytes();q={}
for key in [b'max_flat_workgroup_size',b'.max_flat_workgroup_size',b'.name']:
 positions=[];start=0
 while True:
  i=b.find(key,start)
  if i<0:break
  positions.append({'offset':i,'after_hex':b[i+len(key):i+len(key)+20].hex(),'before':b[max(0,i-100):i].decode(errors='replace')});start=i+len(key)
 q[key.decode()]=positions[:50]
print(json.dumps(q))
