import pathlib,json,hashlib,ast,shutil
from Deep_latent_maps_r8946 import validate,maps
r=pathlib.Path.cwd();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(n,s):
 if n.endswith('.py'):ast.parse(s)
 (r/n).write_bytes(s.encode('utf8'))
def write(n,v):put(n,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rename(s):return s.replace('Deep_latent_prefill_component_r8939','Deep_latent_mapping_component_r8946').replace('deep-latent-prefill-component-r8939','deep-latent-mapping-r8946').replace('Deep-latent-prefill-component-manifest-r8939','Deep-latent-mapping-manifest-r8946')
put('Deep_latent_mapping_component_r8946_inside.sh',rename((r/'Deep_latent_prefill_component_r8939_inside.sh').read_text()))
put('launch_Deep_latent_mapping_component_r8946.py',rename((r/'launch_Deep_latent_prefill_component_r8939.py').read_text()))
audit='''import pathlib,json,hashlib,sys
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['status']=='COMPLETED_DIAGNOSTIC_NO_MODEL_QUALIFICATION'
assert [v['kind']for v in q['fixtures']]==['swa','hca128','csa4'];assert len(q['normal_owners'])==8 and all(v['normal']for v in q['normal_owners'])
for case in q['fixtures']:
 assert len({v['seed']for v in case['owners']})==8 and len({v['operand_sample_SHA256']for v in case['owners']})==8
 assert [(v['output'],v['width'])for v in case['measurements']]==[(0,2048),(1,2048),(2,512),(3,64)]
 for row in case['measurements']:
  assert len(row['owners'])==8
  for v in row['owners']:assert v['CAPI_calls']==2048//row['width']and v['counter_delta']==([1,2048,64]if row['output']==1 else[0,0,0])
 assert set(case['comparisons'])=={'(0, 1)','(0, 2)','(1, 3)'}
print(json.dumps({'passed':True,'actual_handle':q['step'],'normal_owners':8,'fixtures':3,'scope':'Engineeringfinite/count/schema auditonly;numericerrornotModelqualitygate','raw_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
''';put('verify_Deep_latent_mapping_component_r8946.py',audit)
contract=dict(hardware='BW1000',level='CPU_CONSUMER_CONTRACT',status='PASSED',CPU_only=True,no_GPU_dlopen=True,fixtures=validate(),ABI={'symbol':'modern_deep_latent_strided_r681','args':'int+6size_t+8int64+9voidptr','total':24,'mode0_T2048_T512':'currentFP32reference','mode1_T2048':'query32candidate','mode0_T64':'existingfused64 noaddedinterface'},fixed_CAPI_per_owner=114,new_native_build=False,query_pointer_offset_contract='narrow query andselectionrows; originalglobalbanksleftwhole,outputsliceoffset only',checks=['All2048row ns/nc/causal/globalrequestIDranges/duplicate/padding checked CPU','Rejoining32/64/512queryslices reproducesmetadatabytes','Explicit perdevice torch.Generator seed894600+100*d+fixtureid, sampleSHA uniqueness required beforecompute','CPU metadata checks beforeCUDA/dlopen;actualsame DSO5a044/receipt218/objecthash'],stop='Any nativeerror/nonfinite/invalidmap/counter mismatch/RNG sample collision stops fixedpacket andcleanup; numericaldifferencesreportedwithoutbitexactgate, noautomaticModelreload.',boundary_metrics=['maxabs','relativeRMS','BF16ULPmax/mean','tilefirstrowvsothersmaxabs','requestfirstrowmaxabs','perrequestmaxabs','tileedgechangedrowcounts'],Rope_scope='Standardtheta10000onlast64dims,positions0..511;syntheticRMSnormalizedq/KV,notactualYarnlearnedcompressoractivation.')
write('Deep-latent-mapping-CPU-contract-r8946.json',contract)
m=json.loads((r/'Deep-latent-prefill-component-manifest-r8939.json').read_bytes());names=['Deep_latent_maps_r8946.py','Deep_latent_mapping_component_r8946.py','Deep_latent_mapping_component_r8946_inside.sh','launch_Deep_latent_mapping_component_r8946.py','verify_Deep_latent_mapping_component_r8946.py'];m['incoming_source_SHA256']={n:sha(r/n)for n in names};m['scope']='FixedsyntheticRMS/RoPE B4I512 actualSWA128/HCA128/CSA4 globalmaps; fullreference/query32/perrequestreferenceconcat/existingtile64;8independentseededowneroperands. NoModelweightsCCL/newnativebuild.';m['plan']='3fixturesx(1fullreference+1query32+4perrequestreference+32existingtile64)=114CAPI/owner;notgrid/noextraarm/warm/retry';m['CPU']={'reused_finalizer':'910407.218','new_native_compile':False,'reused_DSO':True};m['source_consumer_contract_SHA256']=sha(r/'Deep-latent-mapping-CPU-contract-r8946.json');m['quality']='Engineeringfinite/owner/count/audit/post; numericerrorsdiagnostic,notbitexactgate andnotModeltokenquality.';m['status']='CPU_READY_NO_GPU_AUTHORIZATION';m['memory']={'T2048H8D512_query_MiB':16,'four_outputs_MiB':64,'whole_globalbanks':'SWA2048rows+CSA512rows max, reusethe samecase forfourmethods','peak':'Torchfloatingtemporaries UNKNOWNboundedbyfixedsmallshapes;notModelpeak'};m['budget']={'hard_seconds':300,'inner_seconds':240,'fixed_114CAPI_each_8owners':True,'planning_seconds':100,'margin_seconds':140,'scope':'90517calls~60swholepacketincludingstartup;6fullreference-equivalentdatasetsvs4in905,andexistingtile64smallcalls;conservative100s planningonly,noguarantee/noextension'};write('Deep-latent-mapping-manifest-r8946.json',m)
stage=rename((r/'stage_Deep_latent_prefill_component_r8939.py').read_text());stage=stage.replace("names=['Deep_latent_mapping_component_r8946.py'", "names=['Deep_latent_maps_r8946.py', 'Deep_latent_mapping_component_r8946.py'")
stage=stage.replace('Deep-latent-prefill-component-staged-r8939','Deep-latent-mapping-staged-r8946');put('stage_Deep_latent_mapping_component_r8946.py',stage)
print(json.dumps({'CPU_only':True,'sourcehash':m['incoming_source_SHA256'],'manifest_SHA256':sha(r/'Deep-latent-mapping-manifest-r8946.json'),'calls_per_owner':114,'GPU':'NOT_AUTHORIZED'}))
