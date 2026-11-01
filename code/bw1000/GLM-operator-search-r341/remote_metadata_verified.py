import pathlib,json,hashlib,re
p=pathlib.Path.home()/'hygon-inference-refactor-20261014/candidates/GLM-operator-search-r341/fused.so';b=p.read_bytes();key=b'.max_flat_workgroup_size';start=0;rows=[]
while True:
 i=b.find(key,start)
 if i<0:break
 j=i+len(key);assert b[j]==0xcd;limit=int.from_bytes(b[j+1:j+3],'big');j+=3;assert b[j:j+6]==b'\xa5.name';j+=6;assert b[j]==0xd9;length=b[j+1];name=b[j+2:j+2+length].decode();m=re.search(r'fusedILi(\d+)ELi(\d+)',name);assert m
 lanes,groups=map(int,m.groups());variant=[16,32,64,128].index(lanes)*4+[1,2,4,8].index(groups);rows.append({'name':name,'variant':variant,'requested_threads':lanes*groups,'compiled_max_flat_workgroup_size':limit,'static_launch_valid':lanes*groups<=limit});start=i+len(key)
assert len(rows)==16 and len({x['variant'] for x in rows})==16
print(json.dumps({'library':str(p),'library_SHA256':hashlib.sha256(b).hexdigest(),'kernels':sorted(rows,key=lambda x:x['variant']),'static_INVALID_BOUND_variants':sorted(x['variant'] for x in rows if not x['static_launch_valid']),'GPU_initialized':False}))
