import json,pathlib,sys,statistics,math,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_text());assert q['completed']and len(q['owners'])==8;assert sorted(o['device']for o in q['owners'])==list(range(8))
for o in q['owners']:
 assert o['completed']and o['normal_tensor_cleanup']and o['selected_CPU_oracle_exact']and o['full_output_finite']and o['all_guards_and_input_unchanged'];assert o['blocks']==1280 and o['threads']==256 and o['independent_chains']==8 and o['loops']==65536
 assert o['operations_per_kernel']==2*1280*256*8*65536;assert o['before_output_SHA256']==o['after_output_SHA256'];assert o['CPU_FP64_oracle_selected_outputs']==512
 assert len(o['event_seconds'])==len(o['wall_seconds'])==5 and all(math.isfinite(x)and x>0 for x in o['event_seconds']+o['wall_seconds'])
 assert math.isclose(o['event_TFLOPS_median'],o['operations_per_kernel']/statistics.median(o['event_seconds'])/1e12)
print(json.dumps({'passed':True,'step':q['step'],'normal_owners':8,'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'event_TFLOPS_by_device':[o['event_TFLOPS_median']for o in q['owners']],'scope':q['scope']}))
