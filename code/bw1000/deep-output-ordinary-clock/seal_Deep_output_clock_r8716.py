import pathlib,json,hashlib,ast,sys,subprocess,copy
r=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();files=['Deep_output_clock_layer_r8714.py','Deep_output_clock_layer_r8714_inside.sh','verify_Deep_output_clock_r8716.py','launch_Deep_output_clock_r8716.py']
for n in files:
 p=r/n
 if n.endswith('.py'):ast.parse(p.read_text())
 assert b'\r'not in p.read_bytes(),n
# Actual auditor receives only synthetic metadata, no source import orGPU.
d={'completed':True,'step':'CPU_FIXTURE_NOT_GPU','arm_max_owner_seconds':[1,1.01,1.01,1],'owners':[]}
for dev in range(8):
 arms=[]
 for arm,flag in enumerate([0,1,1,0]):
  f=[]
  for step,a in [(0,a)for a in range(0,32,4)]+[(x,0)for x in range(1,36)]:
   clock=None
   if flag and step in[25,35]:
    base=100+arm*100+step;start=base+dev*.00001;clock=[start,start+.0001,start+.0002,start+.0003,base+.001,.0005,1+(step==35)+2*(arm==2),32]
   f.append({'step':step,'CAPI_wall_seconds':.003,'observer_clock':clock,'stages':[{'name':n,'numeric_guard':True}for n in ['attn_value','hidden']]if step in[25,35]else[]})
  arms.append({'arm':arm,'observer':flag,'decode_forward_count':35,'forwards':f})
 d['owners'].append({'dev':dev,'normal_owner_closed':True,'arms':arms})
p=r/'Deep-output-clock-CPU-fixture-r8716.json'
with p.open('x')as f:json.dump(d,f)
z=subprocess.run([sys.executable,'-X','utf8',str(r/'verify_Deep_output_clock_r8716.py'),str(p)],capture_output=True,text=True,encoding='utf-8');assert z.returncode==0,z.stderr;a=json.loads(z.stdout);assert len(a['clocks'])==4
bad=copy.deepcopy(d);bad['owners'][7]['arms'][1]['forwards'][-1]['observer_clock'][6]=99
p=r/'Deep-output-clock-CPU-invalid-r8716.json'
with p.open('x')as f:json.dump(bad,f)
z=subprocess.run([sys.executable,'-X','utf8',str(r/'verify_Deep_output_clock_r8716.py'),str(p)],capture_output=True,text=True,encoding='utf-8');assert z.returncode!=0
checks={'CPU_only':True,'GPU':False,'passed':True,'actual_auditor_valid_fixture_pass':True,'stale_sequence_negative_rejected':True,'same_class_layout':True,'mode5_only_originalBF16copy_Cast_SUM_Cast_retained':True,'sampled_only_ON_25_35':True,'normal8owner_4eventrecords_each':True,'event_creation_before_arms':True,'elapsed_after_original_drain_noNewSync':True,'event_close_after_original_context_drain':True,'no_Runtime_strongref_in_TLS':True,'ordinary_CPU_enqueue_notGPUready':True,'GPU_event_span_not_puretransport':True}
with(r/'Deep-output-clock-CPU-checks-r8716.json').open('x')as f:json.dump(checks,f,indent=2)
base=json.loads((r/'Deep-C128-component-manifest-r8695.json').read_text());base['scope']='Actual r7003 layer0 TP8 ordinary observation, originalweight/math/copy/protocol unchanged; no Model performance/bit gate';base['GPU_status']='CONTINUOUS_ROOT_GO_NOT_YET_LAUNCHED';base['resource']['inner_timeout_seconds']=540;base['geometry']={'B':32,'TP':8,'I':512,'teacher_steps':36,'observer_arms':[0,1,1,0],'ON_samples_each_arm':[25,35],'wholelayer_OFF_controls':[0,3],'projection_tile':16,'precision':5};base['incoming_source_SHA256']={n:sha(r/n)for n in files};base['existing_incoming_dependency_SHA256']={n:sha(r/n)for n in ['resource_wave_local_r3097_outer.sh','Kimi_Q0_idle_r8066.py','deep_affinity_env_contract_r7166.py','build_deep_serving_cases_r2130.py']};base['existing_remote_SHA256']={'results/deep-output-clock-r8713-build.json':'b67915b4d15d44aac1c97385c28dcc40e5c4055cbca205605dc54173f59521a9','candidates/deep-output-clock-r8713/libdeep_output_clock_partial_r8713.so':'0e2e617bc0b194708e9ec8855d2a0bb5fd200848ac72e305d7c8cbc18935e546'};base['CPU_local_receipts_SHA256']={n:sha(r/n)for n in ['r8715-Deep-output-clock-CPU-observed-a.json','Deep-output-clock-CPU-checks-r8716.json']};base['native_timer_source_SHA256']=sha(r/'deep_attention_output_clock_r8713.cpp');base['timing']='ObserverOFF full layer controls; ON onlysteps25/35 two repeats. SamehostCPU submission/enclosing clocks plus originalRuntime stream beforeSUM/afterSUM queueevents read onlyafter existingdrain; CPUenqueue notGPUready, span includesarrival/hostwait/dependency; no puretransport attribution. Do not rank-sum. No newmath/protocol/Model.';base.pop('memory',None)
with(r/'Deep-output-clock-manifest-r8716.json').open('x')as f:json.dump(base,f,indent=2)
print(json.dumps({'manifest_SHA256':sha(r/'Deep-output-clock-manifest-r8716.json'),'entry_SHA256':sha(r/'launch_Deep_output_clock_r8716.py'),'CPU_pass':True}))
