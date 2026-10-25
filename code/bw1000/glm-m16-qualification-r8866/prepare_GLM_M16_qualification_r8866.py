import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent;old=r/'GLM-M16-Model-private-r8862';fix=r/'GLM-M16-consumer-repair-private-r8865';d=r/'GLM-M16-Model-private-r8866';d.mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=lambda p,x:p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
parent=json.loads((old/'GLM-CROSSREQUEST-Model-source-manifest-r8862.json').read_bytes());oldmh=sha(old/'GLM-CROSSREQUEST-Model-source-manifest-r8862.json')
names=list(parent['source_SHA256'])+['dispatch_GLM_resident_exact_Model_r8862.py']
for n in names:
 raw=(old/n).read_bytes()
 if 'r8862'in n:raw=raw.replace(b'r8862',b'r8866')
 (d/n.replace('r8862','r8866')).write_bytes(raw)
for n in('glm_crossrequest_control','glm_prefill_micro_probe'):
 raw=(fix/(n+'_r8865.py')).read_bytes().replace(b'r8862',b'r8866').replace(b'r8865',b'r8866');(d/(n+'_r8866.py')).write_bytes(raw)
worker=d/'modern_glm_prefill_micro_r8866.py';raw=worker.read_bytes().replace(b'B16thenB32',b'B32M8thenM16');worker.write_bytes(raw)
m=json.loads(json.dumps(parent).replace('r8862','r8866'));m['root_GO']='READY_WAIT_FINAL_ROOT_MODEL_GO';m['scope']='Repair original564 actual Python finish override; same CPU210 native68bdbd and allSDK/math. One load B32M8 thenB32M16/32distinct477/I512O128/prefill8/constructor32/cap640/fixedadmittedexact160MiBmask; prefix/vectorON,PV/tile2/quantreuseOFF;6208MiBadmission/512MiBfloor/drain/noeviction.910376n12+n13/16DCU/c64each/240000Meach/45minhard2600inner;twofullarms only/no retry/extension.'
m['maximum_full128_arms']=2;m['maximum_loads']=1;m['mechanical_revision']={'original564':'FAILED1/actual override rejected correctM16 vector counts;0normal/audit1/post565empty retained','repair':'Only actual finish vector16 expectation, refresh existing last_completed_generation atomic/fsync before finish, success-only completion/QA scope; no nativecompile/newmath','recovered_evidence':'GLM564-saved-recovery-compact-r8865-c.json;16owner fullIDsEOS and realemitter clocks recovered;notqualification/performance round'}
m['diagnostic_budget']='45minhard2600inner. Original564 failed at2108s after both full128 generations; same scope expected to leave492s for final QA/shutdown, varianceUNKNOWN. Exactly2fullarms; no optionalrepeat/retry/extension.'
side=d/'GLM-transient-Model-observer-manifest-r8866.json';s=json.loads(side.read_bytes());s['source_SHA256']={n.replace('r8862','r8866'):sha(d/n.replace('r8862','r8866'))for n in json.loads((old/'GLM-transient-Model-observer-manifest-r8862.json').read_bytes())['source_SHA256']};write(side,s);m['observer_manifest_SHA256']=sha(side)
m['source_SHA256']={n.replace('r8862','r8866'):sha(d/n.replace('r8862','r8866'))for n in parent['source_SHA256']};mp=d/'GLM-CROSSREQUEST-Model-source-manifest-r8866.json';write(mp,m);mh=sha(mp)
ep=d/'dispatch_GLM_resident_exact_Model_r8866.py';raw=ep.read_bytes();assert raw.count(oldmh.encode())==1;ep.write_bytes(raw.replace(oldmh.encode(),mh.encode()));eh=sha(ep)
for n in m['source_SHA256']:
 if n.endswith('.py'):ast.parse((d/n).read_bytes(),filename=n)
for stem in('stage_GLM_M16_Model','invoke_GLM_M16_Model','upload_GLM_M16_Model','observe_GLM_M16_Model'):
 raw=(r/(stem+'_r8862.py')).read_bytes().replace(b'r8862',b'r8866').replace(oldmh.encode(),mh.encode()).replace(hashlib.sha256((old/'dispatch_GLM_resident_exact_Model_r8862.py').read_bytes()).hexdigest().encode(),eh.encode());(r/(stem+'_r8866.py')).write_bytes(raw)
test=(r/'check_GLM564_consumer_repair_r8865.py').read_bytes().replace(b'r8862',b'r8866').replace(b'r8865',b'r8866').replace(b'GLM-M16-consumer-repair-private-r8866',b'GLM-M16-Model-private-r8866');(r/'check_GLM564_consumer_repair_r8866.py').write_bytes(test)
print(json.dumps({'GPU':False,'nativecompile':False,'manifest_SHA256':mh,'entry_SHA256':eh,'DSO_SHA256':m['model_SHA256'],'scope':m['scope']}))
