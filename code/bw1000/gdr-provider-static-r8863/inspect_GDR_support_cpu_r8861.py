import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=r/'candidates/public-shca-layers-r2228/layer3-selected/opt/rccl-shca-net/lib/librccl-net-shca.so'
def call(args):
 q=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'rc':q.returncode,'out':q.stdout,'err':q.stderr[:1000]}
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU static read only; no dlopen/GPU/model/submit','plugin_path':str(p),'plugin_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
n=call(['nm','-D','--defined-only',str(p)])
d['symbols']=[x for x in n['out'].splitlines() if any(t.lower() in x.lower() for t in ['GetProperties','GdrSupport','DmaBufSupport','p2p','ncclNetPlugin'])]
for sym in ['ncclIbGetProperties','ncclIbGdrSupport','nccl_p2p_dmabuf_support']:
 d[sym]=call(['objdump','-d','--disassemble='+sym,str(p)])
s=call(['strings',str(p)])
d['strings']=[x for x in s['out'].splitlines() if any(t.lower() in x.lower() for t in ['GDR','peermem','peer_mem','dmabuf','/sys/kernel'])]
d['sdk_libs']=[]
for p in (r/'install/hygon-kimi-5f91ad7-r3672/lib').iterdir():
 if p.is_file() and any(k in p.name.lower() for k in ['ccl','runtime']):
  d['sdk_libs'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'readelf':call(['readelf','-d',str(p)])})
print(json.dumps(d))
