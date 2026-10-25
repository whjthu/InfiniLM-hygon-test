from pathlib import Path
import json,hashlib,base64
r=Path(__file__).resolve().parent;s=r/'InfiniLM-hygon-test/code/bw1000/deep-scalar-cast-qualified-component';out=r/'Deep-host-clock-private-r9049';out.mkdir(exist_ok=True)
mapping={'Deep_scalar_Cast_component_r9054b.py':'Deep_host_clock_component_r9049.py','Deep_scalar_Cast_component_r9054b_inside.sh':'Deep_host_clock_component_r9049_inside.sh','launch_Deep_scalar_Cast_component_r9054b.py':'launch_Deep_host_clock_component_r9049.py','verify_Deep_scalar_Cast_component_r9054b.py':'verify_Deep_host_clock_component_r9049.py','Deep-scalar-Cast-launch-spec-r9054b.json':'Deep-host-clock-launch-spec-r9049.json','deep-scalar-Cast-r9054b':'deep-host-clock-r9049'}
for old in list(mapping)[:4]:
 text=(s/old).read_text(encoding='utf8')
 for a,b in mapping.items():text=text.replace(a,b)
 if old=='Deep_scalar_Cast_component_r9054b.py':
  text=text.replace('import numpy as np','import numpy as np\nimport resource')
  text=text.replace('tick=time.perf_counter()','usage_before=resource.getrusage(resource.RUSAGE_THREAD);thread_before=time.thread_time();tick=time.perf_counter()')
  text=text.replace('elapsed=time.perf_counter()-tick;after=counts(d);','elapsed=time.perf_counter()-tick;thread_cpu=time.thread_time()-thread_before;usage_after=resource.getrusage(resource.RUSAGE_THREAD);after=counts(d);')
  text=text.replace('seconds=elapsed,counter_delta=delta','seconds=elapsed,thread_CPU_seconds=thread_cpu,thread_CPU_fraction=thread_cpu/elapsed,thread_user_seconds=usage_after.ru_utime-usage_before.ru_utime,thread_system_seconds=usage_after.ru_stime-usage_before.ru_stime,voluntary_context_switches=usage_after.ru_nvcsw-usage_before.ru_nvcsw,involuntary_context_switches=usage_after.ru_nivcsw-usage_before.ru_nivcsw,counter_delta=delta')
  text=text.replace("scope='Current1000mode2", "CPU_clock_limit='OSthreadCPUincludesHIPspin/enqueue;notpureCPUoperatorworkorGPUbusy;timingcallsoutsideCAPIonly,noC++/mathchange',scope='Current1000mode2")
 if old=='verify_Deep_scalar_Cast_component_r9054b.py':text+='\nfor f in q["fixtures"]:\n for row in f["measurements"]:\n  for v in row["owners"]:assert v["thread_CPU_seconds"]>=0 and math.isfinite(v["thread_CPU_fraction"])\n'
 (out/mapping[old]).write_bytes(text.replace('\r\n','\n').encode())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((s/'Deep-scalar-Cast-launch-spec-r9054b.json').read_bytes())
manifest['incoming_source_SHA256']={mapping.get(n,n):sha(out/mapping[n])if n in mapping else h for n,h in manifest['incoming_source_SHA256'].items()}
(out/'Deep-host-clock-launch-spec-r9049.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
payload={p.name:base64.b64encode(p.read_bytes()).decode()for p in out.iterdir()if p.suffix in {'.py','.sh','.json'}}
stage="import pathlib,json,base64,hashlib,subprocess\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014/incoming'\npayload="+repr(payload)+"\nout={}\nfor n,b in payload.items():\n p=r/n;data=base64.b64decode(b)\n if p.exists():assert p.read_bytes()==data,n\n else:p.write_bytes(data)\n out[n]=hashlib.sha256(data).hexdigest()\n if n.endswith('.sh'):subprocess.run(['bash','-n',str(p)],check=True)\nprint(json.dumps({'uploaded':out,'no_job_submitted':True}))\n"
(r/'stage_Deep_host_clock_r9049.py').write_text(stage,encoding='utf8')
(r/'invoke_Deep_host_clock_r9049.py').write_text("import pathlib,subprocess,json\nr=pathlib.Path.home()/'hygon-inference-refactor-20261014'\np=subprocess.run(['python3',str(r/'incoming/launch_Deep_host_clock_component_r9049.py')],capture_output=True,text=True,timeout=20)\nassert p.returncode==0,(p.stdout,p.stderr)\nprint(p.stdout)\n",encoding='utf8')
print(json.dumps({'files':len(payload),'native_math_changed':False,'extra_GPU_sync':False,'GPU_scope':'ONE8DCU n05 parent910374 6min300inner noModel/noWeights'}))
