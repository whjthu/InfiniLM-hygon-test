import pathlib,json,hashlib,base64,zlib
r=pathlib.Path('.');b=r/'Kimi-layer0-TP16-replay-private-r9038';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();old=json.loads((r/'Kimi-crossover-Model-private-r9026/Kimi-crossover-manifest-r9026.json').read_text());sourcefiles={'Kimi_layer0_driver_final_r9038.py':b/'layer0_driver.py','controller_Kimi_layer0_r9038.py':b/'controller_gpu.py','launch_Kimi_layer0_r9038.py':b/'launch_gpu.py','Kimi_layer0_GPU_r9038_inside.sh':b/'GPU_inside.sh'};extra={'sources_SHA256':{n:sha(p)for n,p in sourcefiles.items()},'existing_dependency_SHA256':{n:old['existing_dependency_sha256'][n]for n in ['stage_control_r2386.py','resource_wave_local_r3097_outer.sh','Kimi_Q0_idle_r8066.py']},'WAVE_INSIDE':'Kimi_layer0_GPU_r9038_inside.sh','root_continuous_GO':True,'launch_contract':{'parent':'910375','nodes':['a02r4n08','a02r4n09'],'N':2,'ntasks':2,'cpus_per_node':64,'mem_per_node':'128000M','gres_per_node':'dcu:8','limit':'00:10:00','inner_timeout':540,'ownerwidth':4}}
code=r'''import pathlib,json,hashlib,subprocess,os,traceback
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/Kimi-layer0-TP16-replay-r9038f';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'cached_source_verified_prior':'CPU271 reached onlyproviderassert AFTER every915MBslice equalcheckpoint; reusecache noextraction'}
try:
 m=json.loads((i/'Kimi_layer0_seed_r9038.json').read_text());links=list((r/'results').glob('Kimi-layer0-link-CPU-r9038e-*.json'));assert len(links)==1;link=json.loads(links[0].read_text());assert link['completed'];m.update(link);assert sha(m['native'])==m['native_SHA256'];assert sha(m['norm']['path'])==m['norm']['SHA256'];model=pathlib.Path('/public/home/aclsylqidf/shared/Kimi-K3');idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];specs=[('q_proj.weight',0,768),('k_proj.weight',0,768),('v_proj.weight',0,768),('f_a_proj.weight',-1,0),('f_b_proj.weight',0,768),('b_proj.weight',0,6),('g_proj.weight',0,768),('q_conv1d.weight',0,768),('k_conv1d.weight',0,768),('v_conv1d.weight',0,768),('A_log',0,6),('dt_bias',0,768),('o_norm.weight',-1,0),('o_proj.weight',1,768)];m['weights']={str(rank):{}for rank in range(16)};headers={}
 for key,axis,width in specs:
  fullkey='language_model.model.layers.0.self_attn.'+key;fp=model/idx[fullkey]
  if str(fp)not in headers:
   with fp.open('rb')as f:n=int.from_bytes(f.read(8),'little');h=f.read(n)
   headers[str(fp)]=(h,json.loads(h))
  h,hd=headers[str(fp)];meta=hd[fullkey]
  for rank in range(16):
   shape=meta['shape'][:]
   if axis>=0:shape[axis]=width
   path=r/'candidates/Kimi-layer0-TP16-replay-r9038c'/('rank'+str(rank))/(key+'.bin');size=2 if meta['dtype']=='BF16'else4;count=1
   for d in shape:count*=d
   assert path.stat().st_size==count*size;m['weights'][str(rank)][key]={'path':str(path),'SHA256':sha(path),'shape':shape,'dtype':meta['dtype'],'bytes':count*size,'checkpoint_key':fullkey,'checkpoint':str(fp),'checkpoint_header_SHA256':hashlib.sha256(h).hexdigest(),'checkpoint_shape':meta['shape'],'split_axis':axis,'slice_start':rank*width if axis>=0 else None,'slice_width':width if axis>=0 else None,'A_log_first96_beforeTP':key=='A_log'}
 m['weight_binding_bytes_all16']=sum(x['bytes']for z in m['weights'].values()for x in z.values());assert m['weight_binding_bytes_all16']==57208344*16;original=r/'results/kimi-crossover-r9026-TP16PP2-I1928669zh-8e319d252bb323d5-node2.json';assert sha(original)=='9c0b2fd68291c69379449617f07390b3edd7d4ce651c3df9a0b83f219d0dc83d';report=json.loads(original.read_text());m['recorded_outputs']=[]
 for arm in ['A','C']:
  for pos in [536,537]:
   found=None
   for c in report['crossover_calls']:
    if c['arm']!=arm:continue
    for owner in c['owners']:
     if owner['node']!=0:continue
     for rec in owner['trace_after']['records']:
      if rec['layer']==0 and rec['point']=='attn_output'and pos in rec['positions']:found={'arm':arm,'position':pos,'path':rec['remote_owned_buffer'],'SHA256':sha(rec['remote_owned_buffer']),'row':rec['positions'].index(pos)}
   assert found,(arm,pos);m['recorded_outputs'].append(found)
 m['runtime_dependencies_SHA256']['/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so']=sha(r/'install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so');assert m['runtime_dependencies_SHA256']['/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so']=='76eb65adacc8a6832970991ca276b32562543b7fb7e8dd09cda9f4c269f24809';m['config_SHA256']=sha(model/'config.json');m.update(json.loads((i/'Kimi_layer0_launch_extra_r9038.json').read_text()));m['CPU_binding_producer']=q['producer'];manifest=i/'Kimi-layer0-manifest-r9038.json';assert not manifest.exists();manifest.write_text(json.dumps(m,indent=2));cmd=['python',str(i/'Kimi_layer0_driver_final_r9038.py'),'--cpu-contract',str(manifest)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=100);(b/'frontend.log').write_text(p.stdout+p.stderr);q['frontend_command']=cmd;q['frontend_returncode']=p.returncode;assert p.returncode==0,p.stderr[-6000:];q.update(completed=True,manifest=str(manifest),manifest_SHA256=sha(manifest),frontend=json.loads(p.stdout),native_SHA256=m['native_SHA256'],source_SHA256=m['source_SHA256'],binding_bytes=m['weight_binding_bytes_all16'],provider_expected_SHA256=m['provider_expected_SHA256'],provider_relocation_slots=m['provider_relocation_slots'])
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-layer0-finalCPU-r9038-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
''';code=code.replace('else4','else 4');compile(code,'finalcpu','exec');inside='#!/bin/bash\nset -euo pipefail\nulimit -c 0\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython /refactor/incoming/Kimi_layer0_final_CPU_r9038.py\n';files={n:p.read_text()for n,p in sourcefiles.items()};files.update({'Kimi_layer0_final_CPU_r9038.py':code,'Kimi_layer0_final_CPU_r9038_inside.sh':inside,'Kimi_layer0_launch_extra_r9038.json':json.dumps(extra,indent=2)})
for n,s in files.items():assert chr(13)not in s;(b/n).write_bytes(s.encode())
payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode();remote="""import pathlib,json,zlib,base64,subprocess,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming'
for n,s in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=i/n;assert not p.exists(),n;assert chr(13)not in s;p.write_bytes(s.encode())
for n in ['Kimi_layer0_final_CPU_r9038_inside.sh','Kimi_layer0_GPU_r9038_inside.sh']:subprocess.run(['bash','-n',str(i/n)],check=True)
extra=json.loads((i/'Kimi_layer0_launch_extra_r9038.json').read_text())
for n,h in extra['sources_SHA256'].items():assert hashlib.sha256((i/n).read_bytes()).hexdigest()==h,n
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-layer0-finalCPU-r9038','--output='+str(r/'logs/Kimi-layer0-finalCPU-r9038-%J.log'),'env','WAVE_INSIDE_R2323=Kimi_layer0_final_CPU_r9038_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-layer0-finalCPU-r9038-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args}))
""".replace('PAYLOAD',repr(payload));compile(remote,'invoke','exec');(r/'invoke_Kimi_layer0_final_CPU_r9038.py').write_bytes(remote.encode());print('finalmanifestpath/GOT/native/frontend/source payload ready')
