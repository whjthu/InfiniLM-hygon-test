import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=r/'candidates/public-shca-layers-r2228/layer3-selected/opt/rccl-shca-net/lib/librccl-net-shca.so'
raw=p.read_bytes()
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU static bytes plus two authorized nodes read-only sysfs/proc; no dlopen/GPU/model/submit GPU','plugin_SHA256':hashlib.sha256(raw).hexdigest()}
d['gate_string_literals']={hex(a):raw[a:raw.index(b'\0',a)].decode(errors='replace') for a in [0x2216c,0x22181,0x223c8,0x223f8,0x22428]}
code='import pathlib,socket,json; ps=["/sys/kernel/mm/memory_peers/amdkfd/version","/proc/sys/kernel/osrelease","/proc/sys/kernel/numa_balancing"]; print(json.dumps({"node":socket.gethostname(),"files":{p:{"exists":pathlib.Path(p).exists(),"text":pathlib.Path(p).read_text()[:1000] if pathlib.Path(p).exists() else None} for p in ps}}))'
q=subprocess.run(['srun','--jobid=910371','--overlap','--nodes=2','--ntasks=2','--ntasks-per-node=1','--nodelist=a02r3n03,a02r3n07','--cpus-per-task=1','--mem=64M','--time=00:02:00','python3','-c',code],capture_output=True,text=True,timeout=150)
d['cpu_node_read']={'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr[:3000]}
print(json.dumps(d))
