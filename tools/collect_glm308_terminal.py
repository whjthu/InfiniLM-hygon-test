from pathlib import Path
import json,hashlib,datetime,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
label='GLM-prefill-W2-W4-Model-r308'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
wpath=R/'results'/(label+'-watch.json');w=json.loads(wpath.read_bytes())
assert w['completed'] and w['actual_GPU']=='910376.823'
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,
         watch=w,watch_SHA256=sha(wpath),owners=[])
for entry in w['owner_results']:
 p=Path(entry['path']);assert sha(p)==entry['SHA256'];q=json.loads(p.read_bytes())
 fields=['actual_producer','completed','worker_shutdown_completed','post_close_Torch_liveness',
         'normal_reset_close','native_service_loop_completed','normal_shutdown','library_sha256',
         'full_requests_completed','normal_HTTP_shutdown','cleanup','create_seconds','phase_times']
 item=dict(stage=entry['stage'],rank=entry['rank'],SHA256=sha(p),path=str(p),
           lifecycle={k:q[k] for k in fields if k in q})
 item['requests']=q['service_requests']
 if entry['stage']!=3 or entry['rank']!=0:
  keep=['request_id','pair_index','prefill_control_window','paired_warmup','prompt_tokens',
        'requested_output_budget','generated_natural_EOS','reason','actual_prompt_token_ids_SHA256',
        'generated_token_ids_SHA256','generated_tokens','original_reset_all32_completed',
        'Native_forward_host_summary']
  item['requests']=[{k:r[k] for k in keep if k in r} for r in item['requests']]
 else: out['emitter_other_fields']={k:v for k,v in q.items() if k not in ['service_requests']}
 out['owners'].append(item)
p=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+label+'.json')
assert sha(p)==w['CPU_supervisor_SHA256'];out['supervisor']=json.loads(p.read_bytes())
out['supervisor_SHA256']=sha(p)
folder=R/'results'/('GLM-latest-full-owner-r290-client-'+label)/'paired'
out['responses']=[dict(index=i,SHA256=sha(folder/('response-%02d.wire'%i)),
                        wire=json.loads((folder/('response-%02d.wire'%i)).read_bytes())) for i in range(6)]
out['accounting']=subprocess.run(['sacct','-j','910376.823,910407.645','-n','-P','--format=JobIDRaw,JobName%80,State,ExitCode,Elapsed,NodeList'],capture_output=True,text=True,timeout=15).stdout
p=R/'results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json'
out['frozen_request']=dict(SHA256=sha(p),body=json.loads(p.read_bytes()))
print(json.dumps(out))
