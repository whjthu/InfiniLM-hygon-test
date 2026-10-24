import pathlib,json,hashlib,ast,difflib,subprocess,sys,copy
r=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();cpu=json.loads((r/'r8696-Deep-C128-native-CPU-observed-a.json').read_text());libsha=cpu['build']['library_SHA256'];recsha=cpu['receipt_SHA256']
files=['deep_C128_component_r8695.py','verify_Deep_C128_component_r8695.py','Deep_C128_component_r8695_inside.sh','launch_Deep_C128_component_r8695.py']
for n in files:
 p=r/n;s=p.read_text().replace('PENDING_CPU_SHA',libsha).replace('PENDING_CPU_RECEIPT',recsha);p.write_text(s,newline='\n')
 if p.suffix=='.py':ast.parse(s)
original=(r/'deep_C128_component_bridge_r8686.cpp').read_text();changed=(r/'deep_C128_component_bridge_r8694.cpp').read_text()
for x in ['cudaStreamSynchronize','syncStream()','copy_from','prepare(r','commit(r']:
 assert original.count(x)==changed.count(x),x
assert changed.index('producer(stream);timing_valid_r8694=false;const auto timing_begin')<changed.index('auto batch=c->module->prepare')
assert 'r.syncStream();const auto timing_end_r8694='in changed
assert 'thread_local double timing_seconds_r8694'in changed
(r/'Deep-C128-native-clock-r8694.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='original8686',tofile='nativeclock8694')),newline='\n')
fixture=json.loads((r/'Deep-C128-audit-CPU-fixture-r8688.json').read_text());fixture['library_SHA256']=libsha;arms=['O','N']+['O','N','N','O']*3
for o in fixture['owners']:
 records=[]
 for t,arm in enumerate(arms):
  for past in [575,639]:
   wall=100. if t<2 else 1. if arm=='O'else .8;native=90. if t<2 else .7
   records.append({'trial':t,'arm':arm,'past':past,'next_past':past+1,'warmup':t<2,'C128_boundary':past==639,'native_enclosing_wall_s':wall,'native_inner_wall_s':native,'outer_minus_inner_s':wall-native,'validation':{'finite':True,'components_score_pool_rotated_ring_gather':[{'NRMSE':0,'max_abs':0,'byte_equal':True}for _ in range(5)]}})
 o['records']=records
f=r/'Deep-C128-native-audit-CPU-fixture-r8697.json'
with f.open('x')as h:json.dump(fixture,h)
z=subprocess.run([sys.executable,'-X','utf8',str(r/'verify_Deep_C128_component_r8695.py'),str(f)],capture_output=True,text=True,encoding='utf-8');assert z.returncode==0,z.stderr;d=json.loads(z.stdout)
for p,g in d['paired'].items():
 assert len(g['trials'])==12 and len(g['cycles'])==3
 for c in g['cycles']:assert abs(c['paired_speedup']-1.25)<1e-9 and c['native_paired_speedup']==1.
# Negative result contract must reject invalid native range independently of driver.
fixture['owners'][0]['records'][4]['native_inner_wall_s']=1000
f=r/'Deep-C128-native-audit-CPU-invalid-r8697.json'
with f.open('x')as h:json.dump(fixture,h)
z=subprocess.run([sys.executable,'-X','utf8',str(r/'verify_Deep_C128_component_r8695.py'),str(f)],capture_output=True,text=True,encoding='utf-8');assert z.returncode!=0
checks={'passed':True,'CPU_only':True,'GPU':False,'native_compile_actual_handle':'910407.165','native_compile_terminal':'COMPLETED0/8s','fresh_M_equals_MD':True,'source_sync_prepare_commit_copy_call_counts_unchanged':True,'timer_start_after_original_producer_sync':True,'timer_end_after_original_final_drain':True,'timer_read_outside_timed_CAPI_on_same_owner':True,'fixed_thread_local_slots_no_Runtime_reference':True,'ctypes_timer_signature':'int(double*,size_t), length3=[wall_seconds,sequence,tokens]','warmup_excluded':True,'three_ABBA_12_formal_trials':True,'actual_auditor_CPU_fixture_outer1_25_native1_0_recomputed':True,'invalid_native_range_rejected':True,'GPU_math':'NOT_RUN','timing_scope':'steady_clock still includes CPU/OS scheduling and original copy/drain; not pure GPU time or no-noise claim'}
with(r/'Deep-C128-native-CPU-checks-r8697.json').open('x')as h:json.dump(checks,h,indent=2)
m=json.loads((r/'Deep-C128-component-manifest-r8688.json').read_text());m['GPU_status']='NOT_RUN_AWAIT_ROOT_GO';m['geometry']['formal_arms']=['O','N','N','O']*3;m['geometry']['warmup_arms']=['O','N'];m['incoming_source_SHA256']={n:sha(r/n)for n in files};m['existing_remote_SHA256']={'candidates/deep-C128-component-r8694/libdeep_C128_component_r8694.so':libsha,'candidates/deep-C128-component-r8694/CPU-build.json':recsha,'candidates/deep-C128-weight-cache-r8684/CPU-build.json':'e3d4d7adc0aa9d67b460f0f6cdb6aff85b8b0d4cb053820af0f5c20fe619e28b'};m['CPU_local_receipts_SHA256']={n:sha(r/n)for n in ['r8696-Deep-C128-native-CPU-observed-a.json','Deep-C128-native-CPU-checks-r8697.json']};m['native_timer_source_SHA256']=sha(r/'deep_C128_component_bridge_r8694.cpp');m['timing']='Two simultaneous clocks: original Python CAPI outer wall/threadCPU; new native steady_clock after original producer sync through original final drain, fixed thread_local slot read after return. Same8owner concurrency, warmO/N+3ABBA, both clocks max-owner per temporal trial. Native still includes OS scheduling; no new GPU sync/stream/math. Mixed/noisy effect means PARK; no further retries or Model load.'
with(r/'Deep-C128-component-manifest-r8695.json').open('x')as h:json.dump(m,h,indent=2)
print(json.dumps({'manifest_SHA256':sha(r/'Deep-C128-component-manifest-r8695.json'),'entry_SHA256':sha(r/'launch_Deep_C128_component_r8695.py'),'DSO_SHA256':libsha,'CPU_pass':True}))
