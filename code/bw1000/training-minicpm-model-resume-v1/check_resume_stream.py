"""Exercise the same append implementation on small owned CPU files."""
import hashlib,io,json,tempfile
from pathlib import Path
from resume_stream import append_verified
sha=lambda x:hashlib.sha256(x).hexdigest()
rows=[]
with tempfile.TemporaryDirectory(prefix="minicpm-resume-") as tmp:
    root=Path(tmp);prefix=b"original-prefix";tail=b"remaining-payload";whole=prefix+tail
    cases=[("valid",prefix,tail,len(prefix),len(whole),sha(prefix),sha(whole),False),
           ("corrupt_prefix",b"x"+prefix[1:],tail,len(prefix),len(whole),sha(prefix),sha(whole),True),
           ("stale_size",prefix+b"x",tail,len(prefix),len(whole),sha(prefix),sha(whole),True),
           ("negative_offset",prefix,tail,-1,len(whole),sha(prefix),sha(whole),True),
           ("invalid_total",prefix,tail,len(prefix),len(prefix),sha(prefix),sha(whole),True),
           ("short_stream",prefix,tail[:-1],len(prefix),len(whole),sha(prefix),sha(whole),True),
           ("excess_stream",prefix,tail+b"x",len(prefix),len(whole),sha(prefix),sha(whole),True),
           ("whole_digest",prefix,tail,len(prefix),len(whole),sha(prefix),sha(b"wrong"),True)]
    for name,initial,data,offset,total,pin,full,reject in cases:
        p=root/name;p.write_bytes(initial);failed=False
        try:append_verified(p,io.BytesIO(data),offset,total,pin,full,lambda *args:None)
        except (ValueError,EOFError):failed=True
        assert failed==reject,name
        if name=="valid":assert p.read_bytes()==whole
        if name in ("corrupt_prefix","stale_size","negative_offset","invalid_total"):assert p.read_bytes()==initial
        rows.append({"case":name,"rejected":failed,"retained_prefix":p.read_bytes().startswith(initial)})
result={"normal":1,"rejected":7,"cases":rows,"GPU_execution":False}
with Path(__file__).with_name("shared-helper-CPU-checks.json").open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
