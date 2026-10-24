import hashlib,json,pathlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
launch=json.loads((r/'GLM-PP3-constructor-r8781-launch.json').read_text());run=launch['run']
reports=[]
for stage in range(3):
    for rank in range(8):
        path=r/'results'/f'GLM-PP3-constructor-r8781-{run}-stage{stage}-rank{rank}.json'
        j=json.loads(path.read_text())
        assert j['run']==run and j['stage']==stage and j['rank']==rank
        assert j['completed'] and j['normal_destroy'] and j['normal_shutdown']
        assert not j['weights_IO'] and j['forward_calls']==j['resident_calls']==0 and not j['finished']
        assert j['library_SHA256']=='99b99b3ab56de735cbff177df532af0a7ce4836d6af25ce5109dfc5c997d4862'
        assert not j.get('error') and not j.get('cleanup_error')
        reports.append({'file':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),**j})
assert sorted(j['participant'] for j in reports)==list(range(24))
result={'qualified_constructor_only':True,'Model_performance':False,'weights_IO':False,'resident_or_scratch_peak_proven':False,
        'all24_normal':True,'minimum_actual_free_after_create_trim_CCL':min(j['after_trim_and_CCL_tiny_control']['free'] for j in reports),
        'minimum_actual_free_minus_estimated_resident':min(j['free_minus_estimated_future_bank'] for j in reports),
        'all24_estimated_resident_plus6GiB_capacity_screen':all(j['capacity_screen_only'] for j in reports),'reports':reports}
print(json.dumps(result))
