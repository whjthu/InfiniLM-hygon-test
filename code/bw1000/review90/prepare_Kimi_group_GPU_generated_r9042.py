import pathlib,json,hashlib,ast,base64,zlib
b=pathlib.Path('Kimi-rowreduce-group-private-r9042');old=pathlib.Path('Kimi-layer0-owner-mask-private-r9039');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
s=(old/'layer0_driver.py').read_text().replace('r9039','r9042')
s=s.replace("m['arms']=={'A':[1]*32,'C':[1]*24+[8]}","m['arms']=={'A':[1]*32,'B':[1]*24+[8],'C':[1]*24+[8]}")
s=s.replace("'planned_native_calls_per_rank':59,'planned_native_calls_allranks':944","'planned_native_calls_per_rank':345,'planned_native_calls_allranks':5520")
s=s.replace("'raw_storage':'REMOTE_ONLY'","'raw_storage':'REMOTE_ONLY','timing_contract':'untraced finalW8 run CAPI includes native sync and host copy; one warmup pair then four alternating measured pairs'")
s=s.replace("os.environ['INFINILM_KIMI_KDA_TRACE_LAYER']='0'","os.environ['INFINILM_KIMI_OPROJ_W1_R9040']='0';os.environ['INFINILM_KIMI_OPROJ_TRACE_R9040']='1';os.environ['INFINILM_KIMI_KDA_TRACE_LAYER']='0'")
needle="for width in [512,*widths]:\n     read="
rep="""for width in [512,*widths]:
     if device==0:os.environ['INFINILM_KIMI_OPROJ_W1_R9040']='1'if arm=='C'and width==8 else'0';control.collect('select:'+arm+':'+str(past),{'node':node,'selector':os.environ['INFINILM_KIMI_OPROJ_W1_R9040']})
     barrier.wait()
     read="""
assert s.count(needle)==1;s=s.replace(needle,rep)
needle="   own['normal_close']=lib.layer0_close_r9038(ctx)==0;"
rep="""   # Trace/capture is outside all timed trials. Prefix always uses the unchanged selector-OFF path.
   own['timing_trials']=[]
   for trial in range(10):
    candidate=bool((trial%4 in [1,2]));warmup=trial<2
    if device==0:
     os.environ.pop('INFINILM_KIMI_TRACE_DIR',None);os.environ['INFINILM_KIMI_OPROJ_TRACE_R9040']='0';os.environ['INFINILM_KIMI_OPROJ_W1_R9040']='0';control.collect('benchmark:'+str(trial),{'node':node,'candidate':candidate})
    barrier.wait();check(lib.layer0_reset_r9038(ctx));active,shadow=1,9;past=0
    for width in [512,*([1]*24),8]:
     if device==0:
      os.environ['INFINILM_KIMI_OPROJ_W1_R9040']='1'if candidate and width==8 else'0'
      if width==8:control.collect('benchmark_final:'+str(trial),{'node':node,'candidate':candidate})
     barrier.wait();read=1 if past==0 else active;write=1 if past==0 else shadow;x=np.ascontiguousarray(norm[past:past+width]);guard=np.full(width*7168+64,0x7fc1,np.uint16);out=guard[32:-32].reshape(width,7168);before=hashlib.sha256(x.tobytes()).hexdigest();t=time.perf_counter();check(lib.layer0_run_r9038(ctx,width,read,write,x.ctypes.data,out.ctypes.data));wall=time.perf_counter()-t;assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1);assert not np.any((out&0x7f80)==0x7f80);assert hashlib.sha256(x.tobytes()).hexdigest()==before
     if width==8:own['timing_trials'].append({'trial':trial,'candidate':candidate,'warmup':warmup,'wall_s':wall,'output_SHA256':hashlib.sha256(out.tobytes()).hexdigest(),'trace':False})
     if past>0:active,shadow=shadow,active
     past+=width
    assert past==544;barrier.wait()
   own['normal_close']=lib.layer0_close_r9038(ctx)==0;"""
assert s.count(needle)==1;s=s.replace(needle,rep)
# Extend analysis: A serial reference against each W8 arm, plus B/C direct local/reduced diagnosis.
start=s.index("  aa=[records('A',25)");end=s.index(" report['recorded678_output_diagnostic']=[]",start)
s=s[:start]+"""  aa=[records('A',25),records('A',26)]
  for arm in ['B','C']:
   cc=records(arm,25);cm={z['point']:z for z in cc}
   for j,pos in enumerate([536,537]):
    for rec in aa[j]:
     other=cm.get(rec['point'])
     if other is None:continue
     shape,osh=rec['shape'],other['shape'];dtype=rec['dtype'];is_token=len(shape)>=2 and shape[0:2]==[1,1] and len(osh)==len(shape)and osh[0:2]==[1,8]and shape[2:]==osh[2:];is_headnorm=rec['point']=='normalized_f32'and shape==[6,128]and osh==[48,128]
     if not(is_token or is_headnorm):continue
     rawA=(root/'A'/rec['file']).read_bytes();raw=(root/arm/other['file']).read_bytes();n=len(rawA);report['boundary_comparisons'].append({'rank':node*8+device,'position':pos,'point':rec['point'],'comparison':'A_vs_'+arm,'dtype':dtype,**compare(rawA,raw[j*n:(j+1)*n],dtype)})
  bm={z['point']:z for z in records('B',25)};cm={z['point']:z for z in records('C',25)}
  for point in ['r9042_o_proj_local','r9042_o_proj_reduced','output','gated','kda_output']:
   a,bb=bm[point],cm[point];ra=(root/'B'/a['file']).read_bytes();rb=(root/'C'/bb['file']).read_bytes();n=len(ra)//8
   for j,pos in enumerate([536,537]):report['boundary_comparisons'].append({'rank':node*8+device,'position':pos,'point':point,'comparison':'B_vs_C','dtype':a['dtype'],**compare(ra[j*n:(j+1)*n],rb[j*n:(j+1)*n],a['dtype'])})
"""+s[end:]
s=s.replace("shared_outputs[(0,record['arm'])]","shared_outputs[(0,'B'if record['arm']=='C'else'A')]")
s=s.replace("wall=time.perf_counter()-t;assert", "wall=time.perf_counter()-t;barrier.wait();assert")
s=s.replace("for a in ['A','C']","for a in ['A','B','C'])+sum(len(z['timing_trials'])*26 for z in report['workers']")
s=s.replace("==472","==2760").replace("'native_calls':472","'native_calls':2760").replace("LAYER0_REPLAY_COMPONENT_DIAGNOSTIC_COMPLETE","ROWGROUP_COMPONENT_DIAGNOSTIC_COMPLETE").replace("samegeometry_repeat='NOT_RUN; oneA/Cmatrix only'","samegeometry_repeat='onewarmuppair+fouralternatinguntracedpairs'")
needle=" def check(code):"
groupcheck=''' report['RCCL_group_bindings']={}
 maininfo=Info();delta=next(k for k in m['provider_relocation_slots']if 'KimiK3DeltaAttention'in k);p=ctypes.c_void_p.from_address(native_base+m['provider_relocation_slots'][delta]).value;assert dl.dladdr(ctypes.c_void_p(p),ctypes.byref(maininfo));assert sha(maininfo.file.decode())==m['actual678_DSO_SHA256']
 for name,offset in m['RCCL']['Main_GOT_group_slots'].items():
  address=ctypes.c_void_p.from_address(maininfo.base+offset).value;inf=Info();assert address and dl.dladdr(ctypes.c_void_p(address),ctypes.byref(inf));assert sha(inf.file.decode())==m['RCCL']['SHA256'];report['RCCL_group_bindings'][name]={'path':inf.file.decode(),'SHA256':sha(inf.file.decode()),'method':'actualMainGOT','offset':offset}
'''
assert s.count(needle)==1;s=s.replace(needle,groupcheck+needle)
ast.parse(s);(b/'layer0_driver.py').write_bytes(s.encode())
for name in ['controller_gpu.py','launch_gpu.py','GPU_inside.sh']:
 t=(old/name).read_text().replace('r9039','r9042').replace('==472','==2760').replace('LAYER0_REPLAY_COMPONENT_DIAGNOSTIC_COMPLETE','ROWGROUP_COMPONENT_DIAGNOSTIC_COMPLETE')
 if name.endswith('.py'):ast.parse(t)
 (b/name).write_bytes(t.encode())
cpu='#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\nexport INFINILM_OWNER_CPU_WIDTH_R2392=4\npython /refactor/incoming/Kimi_layer0_driver_r9042.py --cpu-worker-contract /refactor/incoming/Kimi-layer0-manifest-r9042.json\n';(b/'CPU_inside.sh').write_bytes(cpu.encode())
files={'Kimi_layer0_driver_r9042.py':b/'layer0_driver.py','controller_Kimi_layer0_r9042.py':b/'controller_gpu.py','launch_Kimi_layer0_r9042.py':b/'launch_gpu.py','Kimi_layer0_GPU_r9042_inside.sh':b/'GPU_inside.sh','Kimi_layer0_owner_CPU_r9042_inside.sh':b/'CPU_inside.sh'}
receipt=json.loads((b/'CPU-observed-final.json').read_text())['receipt']['data'];fields=['native','native_SHA256','actual678_DSO','actual678_DSO_SHA256','provider_DSO','provider_DSO_SHA256','provider_relocation_slots','provider_expected_SHA256'];extra={k:receipt[k]for k in fields};extra['RCCL']=receipt['RCCL'];extra.update(sources_SHA256={n:sha(p)for n,p in files.items()},WAVE_INSIDE='Kimi_layer0_GPU_r9042_inside.sh',arms={'A':[1]*32,'B':[1]*24+[8],'C':[1]*24+[8]},planned_native_calls_per_rank=345,planned_native_calls_allranks=5520,benchmark={'warmup_pairs':1,'measured_alternating_pairs':4,'trace':False,'selector_only_final_trueW8':True},CPU_build=receipt['producer'],candidate_Main_SHA256=receipt['candidate_Main_SHA256'],root_continuous_GO=True)
payload=base64.b64encode(zlib.compress(json.dumps({'files':{n:p.read_text()for n,p in files.items()},'extra':extra}).encode())).decode()
remote="""import pathlib,json,zlib,base64,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';d=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'Kimi-layer0-manifest-r9039.json').read_text());m.update(d['extra'])
for n,s in d['files'].items():
 p=i/n;assert not p.exists(),n;assert chr(13)not in s;p.write_bytes(s.encode());assert sha(p)==m['sources_SHA256'][n]
for n in ['Kimi_layer0_GPU_r9042_inside.sh','Kimi_layer0_owner_CPU_r9042_inside.sh']:subprocess.run(['bash','-n',str(i/n)],check=True)
p=i/'Kimi-layer0-manifest-r9042.json';assert not p.exists();p.write_text(json.dumps(m,indent=2));args=['srun','--jobid=910375','--overlap','--exact','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w','a02r4n08,a02r4n09','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-layer0-ownerfront-r9042','--output='+str(r/'logs/Kimi-layer0-ownerfront-r9042-%J-%t.log'),'env','WAVE_INSIDE_R2323=Kimi_layer0_owner_CPU_r9042_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-layer0-ownerfront-r9042-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args,'manifest_SHA256':sha(i/'Kimi-layer0-manifest-r9042.json')}))
""".replace('PAYLOAD',repr(payload));compile(remote,'invoke','exec');pathlib.Path('invoke_Kimi_group_front_r9042.py').write_bytes(remote.encode());print('driver AST + LF + 345calls/rank ready')
