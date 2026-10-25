"""Independent CPU audit of one actual online experiment. No token-identity gate."""
import pathlib,json,hashlib,re,subprocess,time
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';name='Kimi-live-TP16PP2-r9046';launch=json.loads((r/(name+'-launch.json')).read_text());run=launch['run'];watch=json.loads((r/(name+'-watch.json')).read_text());m=json.loads((r/'incoming/Kimi-live-taps-r9046/target-manifest-r9046.json').read_text());q={'qualified_runtime':False,'production_adopted':False,'quality_no_regression_proven':False,'run':run,'owners':[]}
try:
    reports=[]
    for node in range(4):
        p=r/'results'/f'kimi-live-r9046-{run}-node{node}.json';d=json.loads(p.read_text());reports.append(d);q['owners'].append({'node':node,'normal8':d.get('all8_owned_workers_shutdown_completed'),'error':d.get('error'),'shutdown_error':d.get('shutdown_error'),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()});assert d['completed']and d['all8_owned_workers_shutdown_completed']and not d.get('error')and not d.get('shutdown_error');assert d['library_sha256']==m['native_DSO_SHA256']and d['serving_protocol_complete']
        cases=d['speculative_quality_cases'];assert any(c['K']==0 and c['useful_decode_tokens']>0 for c in cases)and any(c['K']>0 and c['useful_decode_tokens']>0 for c in cases)
        assert all(c['decode_user_wall_s']>0 and c['native_target_window_s']>0 for c in cases)
        assert all(len(c['generated_ids'])==c['generated_tokens']for c in cases)
    reference=[(c['label'],c['generated_ids'])for c in reports[2]['speculative_quality_cases']]
    assert all([(c['label'],c['generated_ids'])for c in d['speculative_quality_cases']]==reference for d in reports),'same-run transport tokens disagree'
    assert watch['post_KFD_passed']and watch['terminal']['State']=='COMPLETED'and watch['terminal']['ExitCode']=='0:0';handle=watch['terminal']['JobID'];logs=[]
    for node in range(4):
        p=r/'logs'/f'Kimi-live-TP16PP2-{run}-{handle}-{node}.log';raw=p.read_bytes();s=raw.decode(errors='replace');assert 'Using network IBext_v8'in s and 'Using network Socket'not in s;assert not any(re.search(r'rocBLAS.*(?:error|709)|Traceback \(most recent|HIP error|Segmentation fault|terminate called',line,re.I)for line in s.splitlines());logs.append({'node':node,'SHA256':hashlib.sha256(raw).hexdigest()})
    limit=time.monotonic()+45
    while True:
        launch=json.loads((r/(name+'-launch.json')).read_text())
        if launch.get('draft_post'):break
        assert time.monotonic()<limit,'draft cleanup receipt not ready';time.sleep(1)
    assert launch['draft_srun_returncode']==0 and launch['draft_post']['passed']and not launch['draft_post']['kfd_pids'];dp=pathlib.Path(launch['draft_report']);draft=json.loads(dp.read_text());assert draft['status']=='COMPLETED'and draft['normal_close']and draft['cycles'];dlog=r/'logs'/(launch['draft_jobname']+'-'+launch['draft_handle']+'.log');assert dlog.read_text().count('KIMI_DRAFT_LIVE_R9046_NORMAL')==1
    q.update(qualified_runtime=True,all32normal=True,draft_normal=True,draft_result_SHA256=hashlib.sha256(dp.read_bytes()).hexdigest(),draft_handle=launch['draft_handle'],logs=logs,cases=reports[2]['speculative_quality_cases'],scope='True nativeW8greedy onlinecycles with actual trainedDFlash2; native emitterclock notHTTP; GSM8K2 smoke notstandardbenchmark')
except BaseException as e:q['error']=repr(e);raise
finally:
    p=r/'results'/f'{name}-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q))
