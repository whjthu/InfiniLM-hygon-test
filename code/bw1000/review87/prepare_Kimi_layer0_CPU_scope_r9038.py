import pathlib,json,base64,zlib
r=pathlib.Path('.');b=r/'Kimi-layer0-TP16-replay-private-r9038';old=(b/'layer0_native.cpp').read_text();needle='result=c->d->forward(input,fc);';assert old.count(needle)==1;s=old.replace('#include "runtime/runtime.hpp"','#include "runtime/runtime.hpp"\n#include "engine/distributed/kimi_reduce_precision_r3226.hpp"').replace(needle,'infinilm::engine::distributed::reduce_precision_r3226::Scope reduce_scope(*c->r,1);'+needle);(b/'layer0_native_with_scope.cpp').write_bytes(s.encode())
code=r'''import pathlib,json,subprocess,hashlib,os,traceback
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-layer0-TP16-replay-r9038b';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'host_TUs':1,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'purpose':'Preserve originalTextModel B1 reduce_precision Scope around same originalDelta; not GPUqual.'}
try:
 old=json.loads((r/'results/Kimi-layer0-CPU-r9038-910407.267.json').read_text());assert old['completed'];src=b/'layer0_native.cpp';src.write_bytes((r/'incoming/Kimi_layer0_native_scope_r9038.cpp').read_bytes());cmd=old['command'][:];cmd[cmd.index('-shared')+1]=str(src);out=b/'libKimi_layer0_r9038b.so';cmd[cmd.index('-o')+1]=str(out);q['command']=cmd;p=subprocess.run(cmd,text=True,capture_output=True,timeout=150);(b/'build.log').write_text(p.stdout+p.stderr);q['compile_returncode']=p.returncode;assert p.returncode==0,p.stderr[-6000:];nm=subprocess.check_output(['nm','-D',str(out)],text=True);assert 'modern_kimi_reduce_begin_r3226'in nm and 'modern_kimi_reduce_end_r3226'in nm;q.update(completed=True,native=str(out),native_SHA256=sha(out),source_SHA256=sha(src),actual678_DSO=old['actual678_DSO'],actual678_DSO_SHA256=old['actual678_DSO_SHA256'],exports=old['exports'],provider_DSO=old['provider_DSO'],provider_DSO_SHA256=old['provider_DSO_SHA256'],superseded_compile=old['producer'])
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-layer0-CPU-r9038b-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
'''
inside='#!/bin/bash\nset -euo pipefail\nulimit -c 0\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_layer0_CPU_scope_r9038.py\n';files={'Kimi_layer0_native_scope_r9038.cpp':s,'Kimi_layer0_CPU_scope_r9038.py':code,'Kimi_layer0_CPU_scope_r9038_inside.sh':inside}
for n,t in files.items():assert chr(13)not in t;(b/n).write_bytes(t.encode())
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode();remote="""import pathlib,json,zlib,base64,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists(),n;assert chr(13)not in s;p.write_bytes(s.encode())
subprocess.run(['bash','-n',str(i/'Kimi_layer0_CPU_scope_r9038_inside.sh')],check=True)
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-layer0-CPU-r9038b','--output='+str(r/'logs/Kimi-layer0-CPU-r9038b-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_layer0_CPU_scope_r9038_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-layer0-CPU-r9038b-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args}))
""".replace('PAYLOAD',repr(payload));compile(code,'cpu','exec');compile(remote,'invoke','exec');(r/'invoke_Kimi_layer0_CPU_scope_r9038.py').write_bytes(remote.encode())
