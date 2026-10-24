import json,pathlib
q=json.loads(pathlib.Path('Kimi757-phase-r8650-a.json').read_text());w=q['workers'];print({'phase':q['phase'],'normal':sum(v['normal_close']for v in w),'observer_stable':sum(v['observer_stable751']for v in w),'Nca_changes':{str(v['rank']):v['Nca_MLP_target_changed']for v in w},'Ncg_matches':{str(v['rank']):v['Ncg_MLP_target_matchesO']for v in w}})
