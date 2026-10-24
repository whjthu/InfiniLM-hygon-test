import pathlib,json,hashlib,sys,subprocess,math
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';tag=sys.argv[1];p=r/'results'/f'Deep-defer-layer-r7357-{tag}.json';q=json.loads(p.read_text());assert q['completed']and q['no_serving_TPS']and not q['performance_adopted'];assert q['library_SHA256']=='e23d2571c4c9bc40177b616eb145d3e62739cd976c4f77841494f45229f31262'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=json.loads((r/'incoming/Deep-defer-layer-GPU-manifest-r7357.json').read_text())
for n,h in manifest['sources_SHA256'].items():assert sha(r/'incoming'/n)==h
rows=[v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-23','-j',tag,'-n','-P','-o','JobID,State,ExitCode'],text=True).splitlines()];assert len(rows)==1 and rows[0][1:]==['COMPLETED','0:0']
arms=[['validateOFF',0,True],['validateON',1,True],['perfOFF0',0,False],['perfON0',1,False],['perfON1',1,False],['perfOFF1',0,False],['restoreOFF',0,True]];assert q['arms']==arms
owners=q['owners'];assert len(owners)==8 and sorted(o['dev']for o in owners)==list(range(8));masks=[set(o['affinity'])for o in owners];assert all(len(v)==4 for v in masks)and len(set.union(*masks))==32
nonexact=0;maxabs=0.
for o in owners:
 assert o['normal_owner_closed']and len(o['arms'])==7;base={}
 for arm,c in enumerate(o['arms']):
  label,flag,obs=arms[arm];assert c['arm']==arm and c['label']==label and c['defer']==flag and c['validation_observer']==obs and c['decode_forward_count']==35 and len(c['forwards'])==43
  ts=[]
  for z in c['forwards']:
   step,a=z['step'],z['chunk_begin'];assert z['defer_delta']==int(bool(flag and step));assert z['rows']==(list(range(a,a+4))if step==0 else list(range(32)))
   full=bool(obs and step in [1,35]);assert z['full17stage_capture']==full and len(z['stages'])==(17 if full else int(step>0))
   if not obs:assert not full and all(st['name']=='hidden'for st in z['stages'])
   if step:assert math.isfinite(z['CAPI_wall_seconds'])and z['CAPI_wall_seconds']>0;ts.append(z['CAPI_wall_seconds'])
   for s in z['stages']:
    k=(step,s['name']);assert s['numeric_guard']and 0<=s['exact_fraction']<=1
    if arm==0:base[k]=s['SHA256']
    assert s['exact']==(s['SHA256']==base[k])
    if not flag:assert s['exact']
    nonexact+=int(not s['exact']);maxabs=max(maxabs,s['amax'])
  assert math.isclose(sum(ts),c['whole_layer_decode_seconds'],rel_tol=1e-12)
maxima=[max(o['arms'][i]['whole_layer_decode_seconds']for o in owners)for i in range(7)];assert maxima==q['arm_max_owner_seconds']
stepmax=[sum(max(o['arms'][i]['forwards'][8+s]['CAPI_wall_seconds']for o in owners)for s in range(35))for i in range(7)];assert stepmax==q['arm_sum_perstep_max_owner_seconds']
ratio=(stepmax[2]+stepmax[5])/(stepmax[3]+stepmax[4]);assert math.isclose(ratio,q['ABBA_ratio'],rel_tol=1e-12)
spread=max(abs(stepmax[2]-stepmax[5])/((stepmax[2]+stepmax[5])/2),abs(stepmax[3]-stepmax[4])/((stepmax[3]+stepmax[4])/2));gain=ratio-1
out={'completed':True,'Slurm':rows,'normal8owners':True,'quality_component_only':True,'nonexact_stage_comparisons':nonexact,'amax':maxabs,'source_report_SHA256':sha(p),'arm_sum_perstep_max_owner_seconds':stepmax,'arm_max_owner_trajectory_seconds':maxima,'ordinary_ABBA_ratio':ratio,'point_gain':gain,'paired_repeat_spread_fraction':spread,'benefit_exceeds_repeat_spread':gain>spread,'performance_adopted':False,'scope':'TP8 production layer0 teacher35 only, observerOFF ABBA with final existing CAPI drain; hidden fullread after timing. Two separate initial validation and lastrestore arms have17stage snapshots atsteps1/35. NoModel/head/finalIDs/EOS/QA/serviceTPS; no extrapolation43layers.'}
outp=r/'results'/f'Deep-defer-layer-r7357-{tag}-audit.json';assert not outp.exists();outp.write_text(json.dumps(out,indent=2));print(json.dumps(out))
