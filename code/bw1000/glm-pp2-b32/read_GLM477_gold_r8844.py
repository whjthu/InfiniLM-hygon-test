import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
p=r/'results/glm-transient-Model-r5738-rows32-5738-60a66aa1d37a47de9144-stage1-rank0.json'
d=json.loads(p.read_text());q=[c for c in d['multi_batch_cases']if c['phase'].startswith('perf-I512-')]
assert d['completed']and d['worker_shutdown_completed']and len(q)==4
for c in q:assert c['batch']==32 and c['micro']==8 and c['input_lengths']==[512]*32 and c['generated_ids']==q[0]['generated_ids']and c['stopped_at_EOS']==q[0]['stopped_at_EOS']and c['cross_geometry_full_IDs_passed']and c['same_micro_reference_passed']
fp=r/'results/glm-I512x32-real-inputs-r5732.json';f=json.loads(fp.read_text())
assert hashlib.sha256(fp.read_bytes()).hexdigest()=='edfad62bc39efdccf133b4d95ac3b07a44fe76356e013881e67926e017c326fc'
print(json.dumps({'CPU_read_only':True,'source_report':str(p).replace(str(r),'/refactor'),'source_report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'phases':[c['phase']for c in q],'input_fixture':str(fp).replace(str(r),'/refactor'),'input_fixture_SHA256':hashlib.sha256(fp.read_bytes()).hexdigest(),'generated_ids':q[0]['generated_ids'],'stopped_at_EOS':q[0]['stopped_at_EOS'],'input_ids':[c['input_ids']for c in f['cases']]}))
