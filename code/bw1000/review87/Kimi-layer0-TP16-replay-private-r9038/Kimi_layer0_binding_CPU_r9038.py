import pathlib,json,hashlib,subprocess,os,traceback,numpy as np
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/Kimi-layer0-TP16-replay-r9038c';b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
try:
 m=json.loads((i/'Kimi_layer0_seed_r9038.json').read_text());assert sha(m['native'])==m['native_SHA256'];assert sha(m['norm']['path'])==m['norm']['SHA256'];model=pathlib.Path('/public/home/aclsylqidf/shared/Kimi-K3');idx=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];specs=[('q_proj.weight',0,768),('k_proj.weight',0,768),('v_proj.weight',0,768),('f_a_proj.weight',-1,0),('f_b_proj.weight',0,768),('b_proj.weight',0,6),('g_proj.weight',0,768),('q_conv1d.weight',0,768),('k_conv1d.weight',0,768),('v_conv1d.weight',0,768),('A_log',0,6),('dt_bias',0,768),('o_norm.weight',-1,0),('o_proj.weight',1,768)];m['weights']={str(rank):{}for rank in range(16)};headers={}
 for key,axis,width in specs:
  fullkey='language_model.model.layers.0.self_attn.'+key;fp=model/idx[fullkey]
  if str(fp)not in headers:
   with fp.open('rb')as f:n=int.from_bytes(f.read(8),'little');h=f.read(n)
   headers[str(fp)]=(n,h,json.loads(h))
  n,h,hd=headers[str(fp)];meta=hd[fullkey];dtype={'BF16':'<u2','F32':'<f4'}[meta['dtype']];data=np.memmap(fp,dtype=dtype,mode='r',offset=8+n+meta['data_offsets'][0],shape=tuple(meta['shape']))
  if key=='A_log':assert data.shape==(128,);data=data[:96]
  for rank in range(16):
   if axis<0:part=data
   elif axis==0:part=data[rank*width:(rank+1)*width]
   else:part=data[:,rank*width:(rank+1)*width]
   part=np.ascontiguousarray(part);directory=b/('rank'+str(rank));directory.mkdir(exist_ok=True);path=directory/(key+'.bin');part.tofile(path);m['weights'][str(rank)][key]={'path':str(path),'SHA256':sha(path),'shape':list(part.shape),'dtype':meta['dtype'],'bytes':part.nbytes,'checkpoint_key':fullkey,'checkpoint':str(fp),'checkpoint_header_SHA256':hashlib.sha256(h).hexdigest(),'checkpoint_shape':meta['shape'],'split_axis':axis,'slice_start':rank*width if axis>=0 else None,'slice_width':width if axis>=0 else None,'A_log_first96_beforeTP':key=='A_log'}
 m['weight_binding_bytes_all16']=sum(x['bytes']for z in m['weights'].values()for x in z.values());assert m['weight_binding_bytes_all16']==57208344*16
 original=r/'results/kimi-crossover-r9026-TP16PP2-I1928669zh-8e319d252bb323d5-node2.json';assert sha(original)=='9c0b2fd68291c69379449617f07390b3edd7d4ce651c3df9a0b83f219d0dc83d';report=json.loads(original.read_text());m['recorded_outputs']=[]
 for arm in ['A','C']:
  for pos in [536,537]:
   found=None
   for c in report['crossover_calls']:
    if c['arm']!=arm:continue
    for owner in c['owners']:
     if owner['node']!=0:continue
     for rec in owner['trace_after']['records']:
      if rec['layer']==0 and rec['point']=='attn_output'and pos in rec['positions']:
       found={'arm':arm,'position':pos,'path':rec['file'],'SHA256':rec['SHA256'],'row':rec['positions'].index(pos)};break
   assert found,(arm,pos);assert pathlib.Path(found['path']).is_file();assert sha(found['path'])==found['SHA256'];m['recorded_outputs'].append(found)
 symbols=subprocess.check_output(['nm','-D','--defined-only',m['actual678_DSO']],text=True);m['provider_symbols']=[x.split()[-1]for x in symbols.splitlines()if ('KimiK3DeltaAttention'in x and('C1E'in x or'7forwardE'in x))or any(k in x for k in [' modern_kimi_reduce_begin_r3226',' modern_kimi_reduce_end_r3226'])];assert len(m['provider_symbols'])==4,m['provider_symbols'];m['runtime_dependencies_SHA256']['/refactor/install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so']=sha(r/'install/hygon-kimi-5f91ad7-r3672/lib/libinfiniccl.so');m['config_SHA256']=sha(model/'config.json');manifest=b/'binding.json';manifest.write_text(json.dumps(m,indent=2));cmd=['python',str(i/'Kimi_layer0_driver_r9038.py'),'--cpu-contract',str(manifest)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=100);(b/'frontend.log').write_text(p.stdout+p.stderr);q['frontend_command']=cmd;q['frontend_returncode']=p.returncode;assert p.returncode==0,p.stderr[-6000:];q.update(completed=True,manifest=str(manifest),manifest_SHA256=sha(manifest),frontend=json.loads(p.stdout),native_SHA256=m['native_SHA256'],source_SHA256=m['source_SHA256'],binding_bytes=m['weight_binding_bytes_all16'])
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
p=r/'results'/('Kimi-layer0-binding-CPU-r9038-'+q['producer']+'.json');assert not p.exists();p.write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
