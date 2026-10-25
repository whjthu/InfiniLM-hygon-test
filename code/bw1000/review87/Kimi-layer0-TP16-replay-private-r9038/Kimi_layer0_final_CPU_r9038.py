import pathlib,json,hashlib,subprocess,os,traceback
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
   path=r/'candidates/Kimi-layer0-TP16-replay-r9038c'/('rank'+str(rank))/(key+'.bin');size=2 if meta['dtype']=='BF16'else 4;count=1
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
