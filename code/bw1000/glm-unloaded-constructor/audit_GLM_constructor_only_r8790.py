import hashlib,json,pathlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';launch=json.loads((r/'GLM-constructor-only-r8790-launch.json').read_text());run=launch['run'];reports=[]
expected=['world.enter','world.return','TP.enter','TP.return','control.enter','control.return','TextModel.enter','TextModel.return','sync.enter','sync.return','constructor_only.return.unloaded','cleanup.enter','cleanup.return']
for stage in range(3):
    for rank in range(8):
        p=r/'results'/f'GLM-constructor-only-r8790-{run}-stage{stage}-rank{rank}.json';j=json.loads(p.read_text());journal=p.with_suffix('.jsonl');events=[json.loads(v) for v in journal.read_text().splitlines()]
        assert j['completed'] and j['normal_destroy'] and j['normal_shutdown'] and not j.get('error') and not j.get('cleanup_error')
        assert j['Model_constructed'] and not j['weights_IO'] and j['forward_calls']==j['finish_calls']==j['resident_calls']==0
        assert [v['phase'] for v in events]==expected
        assert all(v['stage']==stage and v['rank']==v['device']==rank and v['PID']==j['PID'] and v['participant']==stage*8+rank and v['control_shared_world'] for v in events)
        assert all(a['native_steady_ns']<=b['native_steady_ns'] for a,b in zip(events,events[1:]))
        reports.append({'file':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'journal_SHA256':hashlib.sha256(journal.read_bytes()).hexdigest(),'report':j,'events':events})
assert len({(j['report']['stage'],j['report']['PID']) for j in reports})==24
ids=[j['events'][0]['IDfingerprint_fnv1a64'] for j in reports]
assert len({v['world'] for v in ids})==len({v['control_input'] for v in ids})==1
assert len({v['TP'] for v in ids})==3
assert all(len({reports[stage*8+k]['events'][0]['IDfingerprint_fnv1a64']['TP'] for k in range(8)})==1 for stage in range(3))
print(json.dumps({'qualified_constructor_only':True,'all24_normal':True,'Model_performance_or_resident_peak_proven':False,'all24_estimated_resident_plus_reserve_screen':all(v['report']['estimated_resident_plus_reserve_screen'] for v in reports),'minimum_free_after_trim_CCL':min(v['report']['after_trim_and_tiny_CCL']['free'] for v in reports),'minimum_free_minus_planned_resident':min(v['report']['free_minus_planned_resident'] for v in reports),'reports':reports}))
