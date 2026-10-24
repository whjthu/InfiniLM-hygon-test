import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=r/'candidates/public-shca-layers-r2228/layer3-selected/opt/rccl-shca-net/lib/librccl-net-shca.so'
def call(args):
 q=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'rc':q.returncode,'out':q.stdout,'err':q.stderr[:1000]}
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU static disassembly only; no dlopen/GPU/model/submit','plugin_path':str(p),'plugin_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
for label,start,stop in [('gdr_support','0xf7f0','0xf8f0'),('dmabuf_support','0xf8f0','0xf940'),('ib_get_properties','0xf940','0xfb80'),('get_properties_v8','0x10990','0x109b0')]:
 d[label]=call(['objdump','-d','--start-address='+start,'--stop-address='+stop,str(p)])
s=call(['strings','-t','x',str(p)])
d['strings']=[x for x in s['out'].splitlines() if any(t.lower() in x.lower() for t in ['GDR','peermem','peer_mem','dmabuf','/sys/kernel'])]
print(json.dumps(d))
