"""Local immutable source/CPU closure only; does not submit work."""
import pathlib,hashlib,json,ast,subprocess,sys,copy
r=pathlib.Path(__file__).parent;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files=['deep_C128_component_r8688.py','verify_Deep_C128_component_r8688.py','Deep_C128_component_r8688_inside.sh','launch_Deep_C128_component_r8688.py']
for n in files:
 if n.endswith('.py'):ast.parse((r/n).read_text())
 else:assert b'\r'not in(r/n).read_bytes()
# Exercise the actual independent verifier with synthetic metadata only, including huge excluded warm times.
fixture={'completed':True,'all8normal':True,'all8normal_close':True,'library_SHA256':'7998ae1c3891ce7f4a4737a0ab4648cb1b926398d067e86dd5d03064797eb2e2','actual_handle':'CPU_FIXTURE_NOT_GPU','close':[{'device':i,'normal_close':True}for i in range(8)],'owners':[]}
arms=['O','N']+['O','N','N','O','O','N','N','O']
for dev in range(8):
 records=[]
 for t,arm in enumerate(arms):
  for p in [575,639]:
   v={'finite':True,'components_score_pool_rotated_ring_gather':[{'NRMSE':0,'max_abs':0,'byte_equal':True}for _ in range(5)]}
   records.append({'trial':t,'arm':arm,'past':p,'next_past':p+1,'warmup':t<2,'C128_boundary':p==639,'native_enclosing_wall_s':100 if t<2 else 1 if arm=='O'else .8,'validation':v})
 fixture['owners'].append({'device':dev,'normal':True,'minimum_sampled_free':8*1024**3,'load':[{'mode':i,'free_after':8*1024**3}for i in [0,1]],'records':records})
p=r/'Deep-C128-audit-CPU-fixture-r8688.json'
with p.open('x')as f:json.dump(fixture,f)
z=subprocess.run([sys.executable,str(r/files[1]),str(p)],capture_output=True,text=True);assert z.returncode==0,z.stderr
a=json.loads(z.stdout)
for g in a['paired'].values():
 assert len(g['trials'])==8 and all(x['trial']>=2 for x in g['trials'])
 assert all(abs(c['paired_speedup']-1.25)<1e-9 for c in g['cycles'])
bad=copy.deepcopy(fixture);bad['owners'][7]['records'].pop();p=r/'Deep-C128-audit-CPU-missing-row-r8688.json'
with p.open('x')as f:json.dump(bad,f)
z=subprocess.run([sys.executable,str(r/files[1]),str(p)],capture_output=True,text=True);assert z.returncode!=0
audit={'CPU_only':True,'GPU':False,'passed':True,'synthetic_metadata_only':True,'actual_verifier_executed':True,'warmup_excluded_1_25_speedup_recomputed':True,'missing_record_rejected':True,'numeric_GPU_gate':'NOT_RUN'}
with(r/'Deep-C128-audit-CPU-regression-r8688.json').open('x')as f:json.dump(audit,f,indent=2)
m={'scope':'Actual C128 checkpoint weights + synthetic BF16 hidden compressor component only; no Model/tokens/TPS claim','GPU_status':'NOT_RUN_AWAIT_ROOT_GO','resource':{'parent':'910374','node':'a02r4n05','DCU':8,'CPU':32,'memory_MiB':64000,'hard_minutes':10,'inner_timeout_seconds':480},'geometry':{'rows':32,'dim':512,'ratio':128,'compressed_capacity_per_row':8,'seed_prefix':575,'legal_append_range_inclusive':[575,639],'selected_past':[575,639],'warmup_arms':['O','N'],'formal_arms':['O','N','N','O','O','N','N','O']},'memory':{'immutable_F32_extra_per_C128_GPU_bytes':16777216,'synthetic_history_GPU_bytes':640*32*4096*2,'two_original_contexts_same_owner_Runtime':True,'admit_free_bytes':6*1024**3,'runtime_sampled_floor_bytes':512*1024**2,'sampled_not_peak':True,'fullModel_fit':'NOT_ASSESSED'},'incoming_source_SHA256':{n:sha(r/n)for n in files},'existing_incoming_dependency_SHA256':{n:sha(r/n)for n in ['resource_wave_local_r3097_outer.sh','Kimi_Q0_idle_r8066.py']},'existing_remote_SHA256':{'candidates/deep-C128-component-r8686/libdeep_C128_component_r8686.so':'7998ae1c3891ce7f4a4737a0ab4648cb1b926398d067e86dd5d03064797eb2e2','candidates/deep-C128-component-r8686/CPU-build.json':'a93e8c173cc89ab81a9eec0568a9e3d8df53218d53279726a424b2e3483645ce','candidates/deep-C128-weight-cache-r8684/CPU-build.json':'e3d4d7adc0aa9d67b460f0f6cdb6aff85b8b0d4cb053820af0f5c20fe619e28b'},'CPU_local_receipts_SHA256':{n:sha(r/n)for n in ['r8685-Deep-C128-CPU-observed-a.json','r8687-Deep-C128-component-CPU-observed-a.json','Deep-C128-component-CPU-checks-r8688.json','Deep-C128-audit-CPU-regression-r8688.json']},'timing':'Native enclosing CAPI wall incl original output copies/producer sync/drain; Torch preproducer sync outside timer, reset/seed/readback outside. Slowest of8owners per trial, 2paired ABBA cycles; initialization extra sync reported separately. No extrapolation from bytes/trace to Model gain.'}
with(r/'Deep-C128-component-manifest-r8688.json').open('x')as f:json.dump(m,f,indent=2)
print(json.dumps({'manifest_SHA256':sha(r/'Deep-C128-component-manifest-r8688.json'),'entry_SHA256':sha(r/'launch_Deep_C128_component_r8688.py'),'CPU_audit_fixture_pass':True,'GPU':'NOT_RUN'}))
