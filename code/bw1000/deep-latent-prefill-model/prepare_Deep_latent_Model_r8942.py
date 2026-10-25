import pathlib,json,hashlib,ast
r=pathlib.Path.cwd()
def read(n):return (r/n).read_text(encoding='utf8')
def write(n,s):
 if n.endswith('.py'):ast.parse(s)
 (r/n).write_bytes(s.replace('\r\n','\n').encode('utf8'))
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
old='Deep_owner_Model';new='Deep_latent_Model'
def names(s):return s.replace('Deep-owner-events-Model-manifest-r8925','Deep-latent-Model-manifest-r8942').replace('deep-owner-events-model-r8925','deep-latent-model-r8942').replace('Deep_owner_Model','Deep_latent_Model').replace('r8925','r8942')
s=names(read('modern_Deep_owner_Model_r8925.py'))
s=once(s,"'deep_owner_arm_r8900','deep_owner_read_r8900'","'deep_latent_prefill_mode_r8933','deep_latent_prefill_counts_r8933'")
s=once(s,"report['fused_scope']='AcceptedA PACKON/quant1OFF/activation-scatter-bankleaseOFF. ReusedCPU211 DeviceGuard event collector; math/Runtime/SDK unchanged. Onefull128, firstT2048+decode64 windows only; no performance adoption.'","report['fused_scope']='A/C/C/A full128: acceptedPACKON/quant1OFF/activation-scatter-bankleaseOFF; onlylatentprefillquerytiles32OFF/ON,decodeunchanged; reusedDSO8936,noownercapture.'")
write('modern_Deep_latent_Model_r8942.py',s)
s=names(read('Deep_owner_Model_protocol_r8925.py'))
s=once(s,'"""Single current-r194 metadata factor, one load, fixed B32M32 O/N/N/O."""','"""One load accepted A/C/C/A; only default-OFF latent-prefill query tiles32."""')
s=once(s,"scope='ActualTP8PP1 B32I512O128 Model one load,acceptedPACKON A only owner-event diagnostic. Originalquant2/GemmEx/Runtime/stream/drain;noactivation/scatter/bankleasechange. Oneprimaryfull128 freshacceptedA againstfrozen814;firstprefillT2048+decode64 markedonly,restoreQA,noextraforward/arm'","scope='ActualTP8PP1 B32I512O128 one load,A/C/C/A full128. AcceptedPACKON q1/activation/scatter/bankleaseOFF; onlyprefill latentquerytiles32 OFF/ON. Decodeunchanged; frozen814 andfreshA finalIDs/EOS;originalbefore/restoreQA;noownerextraeventcapture.'")
s=once(s,"def select(arm):check(lib.deep_model_quant1_mode_r8886(arm));g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'","def select(arm):\n  check(lib.deep_model_quant1_mode_r8886(0));check(lib.deep_latent_prefill_mode_r8933(arm));g['os'].environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']='1'")
a=s.index(' lib.deep_owner_arm_r8900.argtypes=');b=s.index(' fresh_A=None',a)
s=s[:a]+''' lib.deep_latent_prefill_mode_r8933.argtypes=[ctypes.c_int];lib.deep_latent_prefill_mode_r8933.restype=ctypes.c_int
 lib.deep_latent_prefill_counts_r8933.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_latent_prefill_counts_r8933.restype=ctypes.c_int
 def latentcounts():
  ans=[]
  for d in range(8):
   q=(ctypes.c_uint64*3)();check(lib.deep_latent_prefill_counts_r8933(d,q,3));ans.append(list(q))
  return ans
 report['owner_event_windows']=[]
 report['owner_event_capture_enabled']=False
'''+s[b:]
s=once(s,'flag=1;select(arm);model_before=modelcounts()','flag=1;select(arm);model_before=modelcounts();latent_before=latentcounts();g[\'checkpoint\'](\'arm_enter_\'+label)')
s=once(s,"report['owner_trials_r8942'].append(entry);entry['pack_mode']=flag;entry['model_arm']=arm","report['owner_trials_r8942'].append(entry);entry['pack_mode']=flag;entry['model_arm']=0;entry['latent_arm']=arm;entry['latent_prefill_chunk_checks']=[]")
a=s.index('    selected=1 if step==0');b=s.index("    entry['CAPI_windows'].append",a)
s=s[:a]+'''    runtime_floor()
    lb=latentcounts()if step==0 else None
    if step==0:g['checkpoint']('prefill_CAPI_enter_'+label,a)
    tick=time.perf_counter_ns();code=lib.modern_native_engine_forward(ctx,sum(sizes),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data));ended=time.perf_counter_ns()
    if step==0:report['selected_native_returncode']=code;g['checkpoint']('prefill_CAPI_return_'+label,a)
    stamps[step,rows]=ended/1e9-began;check(code);runtime_floor()
    if step==0:
     ld=[[y-x for x,y in zip(p,q)]for p,q in zip(lb,latentcounts())];expected=[43,43*sum(sizes),43*((sum(sizes)+31)//32)]if arm else[0,0,0]
     assert ld==[expected]*8,('latent_actual_Tblock_path',arm,sum(sizes),ld,expected)
     entry['latent_prefill_chunk_checks'].append(dict(chunk=a,total_tokens=sum(sizes),counter_delta=ld))
'''+s[b:]
s=once(s,"assert v[arm]==344 and sum(v[:2])==344 and v[2+arm]==pc[3] and sum(v[2:4])==pc[3]\n   assert v[4]==(344 if arm else 0) and v[5]==(43*32*512 if arm else 0) and v[6]==(pc[3] if arm else 0) and v[7]==(43*32*512*6 if arm else 0)","assert v[0]==344 and v[1]==0 and v[2]==pc[3] and v[3:]==[0]*5\n  entry['latent_counter_delta']=[[y-x for x,y in zip(p,q)]for p,q in zip(latent_before,latentcounts())]\n  assert entry['latent_counter_delta']==[[344,43*32*512,344*64]if arm else[0,0,0]]*8,('prefill_only_no_decode_hits',entry['latent_counter_delta'])")
s=once(s,"memory(label+'-after');save();print", "memory(label+'-after');g['checkpoint']('arm_return_'+label);save();print")
a=s.index(" policy(16);entry=one(0,'A-owner-events')")
s=s[:a]+''' policy(16)
 for arm,label in [(0,'A0'),(1,'C1'),(1,'C2'),(0,'A3')]:
  entry=one(arm,label)
  if not entry['complete']:save();raise RuntimeError(label+'_full128_quality_failed_STOP')
 policy(0);select(0);g['run_chat']('after_restore');assert report['finite_after_restore_passed']
 assert report['owner_event_windows']==[]
 report.update(outcome='FULL_ACCA_COMPLETE_PENDING_TERMINAL',quality_passed=True,performance_qualified=True,performance_adopted=False,qualification_scope='fourfull128 frozen814/freshA finalIDs/EOS;before/restore12QA;all8latentTblockcounterchecks',performance_scope='Ordinary unmarkedModel A/C/C/A; nativeprefill/nativeDecode vs driverdecode/E2E andrequestTTFT/ITL separate. Decode sourceunchanged; drift andpairing mustbe reported. Noautomaticadoption.');save()
'''
write('Deep_latent_Model_protocol_r8942.py',s)
for a,b in [('Deep_owner_Model_r8925_inside.sh','Deep_latent_Model_r8942_inside.sh'),('launch_Deep_owner_Model_r8925.py','launch_Deep_latent_Model_r8942.py')]:
 s=names(read(a)).replace('AcceptedAonly-owner-events','latentprefill-only-ACCA').replace('AcceptedAonly','latentprefill-only-ACCA');write(b,s)
s=names(read('verify_Deep_owner_Model_r8925.py'))
s=once(s,"arm=c['model_arm'];assert v[arm]==344 and sum(v[:2])==344 and v[2+arm]==pc[3]and sum(v[2:4])==pc[3];assert v[4]==(344 if arm else 0)and v[5]==(43*32*512 if arm else 0)and v[6]==(pc[3]if arm else 0)and v[7]==(43*32*512*6 if arm else 0)","assert c['model_arm']==0 and v[0]==344 and v[1]==0 and v[2]==pc[3]and v[3:]==[0]*5")
a=s.index("assert not q['performance_qualified']");b=s.index("assert q['minimum_runtime_boundary_free_bytes']",a)
s=s[:a]+'''assert q['performance_qualified']and q['quality_passed']and len(trials)==4 and all(c['complete']for c in trials)
assert [c['latent_arm']for c in trials]==[0,1,1,0]and[c['label']for c in trials]==['A0','C1','C2','A3']
assert q['owner_event_windows']==[]and not q['owner_event_capture_enabled']
for c in trials:
 arm=c['latent_arm'];assert c['latent_counter_delta']==[[344,704512,22016]if arm else[0,0,0]]*8
 assert [v['chunk']for v in c['latent_prefill_chunk_checks']]==list(range(0,32,4))
 for v in c['latent_prefill_chunk_checks']:assert v['total_tokens']==2048 and v['counter_delta']==[[43,88064,2752]if arm else[0,0,0]]*8
'''+s[b:]
write('verify_Deep_latent_Model_r8942.py',s)
s=names(read('check_Deep_guard_Model_error_r8925.py')).replace('Deep-guard-Model-error-CPU-r8942','Deep-latent-Model-error-CPU-r8942')
write('check_Deep_latent_Model_error_r8942.py',s)
print(json.dumps({'prepared':'r8942','GPU':'NOT_AUTHORIZED','new_native_compile':False}))
