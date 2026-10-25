import pathlib,json,hashlib,ast,base64,zlib
r=pathlib.Path('.');b=r/'Kimi-norm-real-input-private-r9037';old=r/'Kimi-gate-real-input-private-r9032';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();obs=json.loads((b/'CPU-observed-a.json').read_text(encoding='utf8'));cpu=obs['receipt']['data'];assert cpu['completed'];(b/'CPU-qualified.json').write_text(json.dumps(cpu,indent=2),encoding='utf8');prior=json.loads((old/'GPU-manifest-local.json').read_text());m={**cpu,'hardware':'BW1000','level':'NATIVE_PRIMITIVE_COMPONENT','status':'CPU266_PASS_UNIQUE_ROOT_NORM_GPU_GO','runtime_dependencies_SHA256':prior['runtime_dependencies_SHA256'],'idle_helper_SHA256':prior['idle_helper_SHA256']};m['provider_symbols']=[n for n in cpu['provider_symbols']if 'forward_inplace'not in n];norms=json.loads((r/'Kimi-projection-real-input-private-r9030/CPU-qualified.json').read_text())['fixture']['records'];m['recorded_norms']=[next(z for z in norms if z['position']==p and z['arm']=='A')for p in [536,537]]
driver=r'''import pathlib,json,hashlib,ctypes,numpy as np,time,os,traceback
r=pathlib.Path('/refactor');m=json.loads((r/'incoming/Kimi-norm-manifest-r9037.json').read_text());job=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('Kimi-norm-real-r9037-'+job+'.json');assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest();q={'status':'FAILED','normal_close':False,'CPU_binding_step':m['producer'],'scope':'Real544tokens/213checkpointembedding rows, originalEmbedding+RMSNorm, original residualcapacity5stride; noDelta/KDA/o_proj/CCL/Model','GPU_F64':False};ctx=None;failure=None
try:
 assert sha(m['native'])==m['native_SHA256'];f=m['fixture'];spec=f['weights'];ew=spec['language_model.model.embed_tokens.weight'];nw=spec['language_model.model.layers.0.input_layernorm.weight'];assert sha(ew['path'])==ew['SHA256']and sha(nw['path'])==nw['SHA256'];w=np.fromfile(ew['path'],dtype='<u2').reshape(213,7168);n=np.fromfile(nw['path'],dtype='<u2');ids=np.array(f['compact_ids'],dtype=np.int64);assert len(ids)==544 and f['residual_capacity']==5;original=[f['unique_token_ids'][i]for i in ids];assert original==f['original_ids'];wh,nh,ih=map(digest,[w,n,ids]);lib=ctypes.CDLL(m['native']);lib.norm_error_r9037.restype=ctypes.c_char_p;lib.norm_create_r9037.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double,ctypes.c_size_t];lib.norm_create_r9037.restype=ctypes.c_void_p;lib.norm_run_r9037.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];lib.norm_readonly_r9037.argtypes=lib.norm_close_r9037.argtypes=[ctypes.c_void_p];lib.norm_counts_r9037.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]
 class Info(ctypes.Structure):_fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];q['actual_bindings']={}
 for name in ['norm_create_r9037','norm_run_r9037','norm_close_r9037',*m['provider_symbols']]:
  info=Info();assert dl.dladdr(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info));p=pathlib.Path(info.file.decode());q['actual_bindings'][name]={'path':str(p),'SHA256':sha(p)}
 for name in m['provider_symbols']:assert q['actual_bindings'][name]['SHA256']==m['provider_DSO_SHA256'],name
 ctx=lib.norm_create_r9037(w.ctypes.data,n.ctypes.data,f['eps'],f['residual_capacity']);assert ctx,lib.norm_error_r9037();assert lib.norm_readonly_r9037(ctx)==0,lib.norm_error_r9037();walls=[]
 def run(a):
  a=np.ascontiguousarray(a);before=digest(a);eg=np.full(len(a)*7168+64,0x7fc1,np.uint16);ng=eg.copy();e=eg[32:-32].reshape(len(a),7168);y=ng[32:-32].reshape(len(a),7168);t=time.perf_counter();assert lib.norm_run_r9037(ctx,len(a),a.ctypes.data,e.ctypes.data,y.ctypes.data)==0,lib.norm_error_r9037();walls.append(time.perf_counter()-t);assert digest(a)==before;assert all(np.all(g[:32]==0x7fc1)and np.all(g[-32:]==0x7fc1)for g in [eg,ng]);assert np.array_equal(e,w[a]),'originalEmbedding row lookup mismatch';assert not np.any((y&0x7f80)==0x7f80);return y.copy()
 serial=[]
 for i in range(544):
  a=run(ids[i:i+1]);again=run(ids[i:i+1]);assert np.array_equal(a,again),'W1 ownrepeat';serial.append(a[0])
 serial=np.stack(serial);prefill=run(ids[:512]);again=run(ids[:512]);assert np.array_equal(prefill,again),'M512 ownrepeat';chunk=run(ids[536:544]);again=run(ids[536:544]);assert np.array_equal(chunk,again),'W8 ownrepeat'
 def diff(a,b):
  af=(a.astype(np.uint32)<<16).view(np.float32).astype(np.float64);bf=(b.astype(np.uint32)<<16).view(np.float32).astype(np.float64);d=af-bf;return {'BF16_different_elements':int(np.count_nonzero(a!=b)),'max_abs':float(np.max(abs(d))),'relative_RMS':float(np.sqrt(np.sum(d*d)/max(float(np.sum(af*af)),1e-30))),'a_SHA256':digest(a),'b_SHA256':digest(b)}
 q['prefill512_vs_serial512']=diff(prefill,serial[:512]);q['W8_vs_serial_true536_543']=[{'position':536+j,**diff(serial[536+j],chunk[j])}for j in range(8)];q['recorded_678_norm_binding']=[]
 for record in m['recorded_norms']:
  assert sha(record['path'])==record['SHA256'];ref=np.fromfile(record['path'],dtype='<u2').reshape(-1,7168)[record['row']];p=record['position'];q['recorded_678_norm_binding'].append({'position':p,'reference_SHA256':record['SHA256'],'W1':diff(ref,serial[p]),'W8':diff(ref,chunk[p-536])})
 q['fixture_binding_536537']=all(z[k]['BF16_different_elements']==0 for z in q['recorded_678_norm_binding']for k in ['W1','W8']);q['all_norm_widths_same']=q['prefill512_vs_serial512']['BF16_different_elements']==0 and all(z['BF16_different_elements']==0 for z in q['W8_vs_serial_true536_543']);actual=np.concatenate([prefill,serial[512:536],chunk]);q['remote_raw_norms']={}
 for name,data in [('A_M512_then32W1',np.concatenate([prefill,serial[512:]])),('C_M512_then24W1_thenW8',actual)]:
  p=r/'results'/('Kimi-norm-'+job+'-'+name+'.bf16');data.tofile(p);q['remote_raw_norms'][name]={'path':str(p),'SHA256':sha(p),'shape':[544,7168],'dtype':'BF16'}
 assert lib.norm_readonly_r9037(ctx)==0,lib.norm_error_r9037();assert [digest(a)for a in [w,n,ids]]==[wh,nh,ih];counts=(ctypes.c_uint64*4)();assert lib.norm_counts_r9037(ctx,counts)==0;q['counts_M512_M1_M8_rows']=list(counts);assert list(counts)==[2,1088,2,2128];assert lib.norm_close_r9037(ctx)==0,lib.norm_error_r9037();ctx=None;q.update(normal_close=True,ownrepeat=True,finite=True,canary=True,host_input_readonly=True,GPU_weight_readonly=True,embedding_lookup_exact=True,wholechain_callwall_s=sum(walls),native_calls=len(walls),status='NORM_RECONSTRUCTION_COMPONENT_DIAGNOSTIC_COMPLETE',Model_finalIDs='NOT_TESTED',causal_Model_conclusion='NOT_PROVEN')
 maps=pathlib.Path('/proc/self/maps').read_text();libs=sorted({x.split()[-1]for x in maps.splitlines()if any(n in x for n in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so'])});q['actual_runtime_dependencies_SHA256']={p:sha(p)for p in libs}
except BaseException as e:failure=e;q.update(exception=repr(e),traceback=traceback.format_exc())
finally:
 if ctx:
  try:lib.norm_close_r9037(ctx);q['normal_close_after_failure']=True
  except BaseException as e:q['close_error']=repr(e)
 q['source_SHA256']=sha(__file__);out.write_text(json.dumps(q,indent=2));print(json.dumps(q))
if failure:raise failure
'''
ast.parse(driver);(b/'norm_driver.py').write_bytes(driver.encode())
# Reuse the exact1009 controller/launch protocol. Each literal changed here is asserted once.
def change(s,a,z,expected=1):assert s.count(a)==expected,(a,s.count(a));return s.replace(a,z)
launcher=(old/'launch_gpu.py').read_text();controller=(old/'controller_gpu.py').read_text()
for var,text in [('launcher',launcher),('controller',controller)]:
 for a,z in [("tag='Kimi-gate-real-r9032'","tag='Kimi-norm-real-r9037'"),("'Kimi-gate-manifest-r9032.json'","'Kimi-norm-manifest-r9037.json'")]:text=change(text,a,z,2 if var=='launcher' and 'manifest' in a else 1)
 if var=='launcher':
  for a,z in [("'Kimi-gate-rootGO-r9032.json'","'Kimi-norm-rootGO-r9037.json'"),("'--job-name=Kimi-gate-idle-r9032'","'--job-name=Kimi-norm-idle-r9037'"),("wave='Kimi_gate_GPU_r9032_inside.sh'","wave='Kimi_norm_GPU_r9037_inside.sh'"),("'controller_Kimi_gate_r9032.py'","'controller_Kimi_norm_r9037.py'")]:text=change(text,a,z)
 else:
  text=change(text,"assert d['status']=='GATE_PROJECTION_COMPONENT_DIAGNOSTIC_COMPLETE'and d['normal_close']and d['counts_all']==[16,10,96]","assert d['status']=='NORM_RECONSTRUCTION_COMPONENT_DIAGNOSTIC_COMPLETE'and d['normal_close']and d['counts_M512_M1_M8_rows']==[2,1088,2,2128]")
  text=change(text,"assert len(d['contexts'])==4 and all(e['closed_normally']and all(e[k]for k in ['ownrepeat','finite','canary','host_input_readonly','GPU_weight_readonly'])for e in d['contexts'].values())","assert all(d[k]for k in ['ownrepeat','finite','canary','host_input_readonly','GPU_weight_readonly','embedding_lookup_exact'])")
  text=change(text,"'--job-name=Kimi-gate-post-r9032'","'--job-name=Kimi-norm-post-r9037'")
 ast.parse(text);(b/('launch_gpu.py'if var=='launcher'else'controller_gpu.py')).write_bytes(text.encode())
inside=(old/'GPU_inside.sh').read_text();inside=change(inside,'incoming/Kimi-gate-manifest-r9032.json','incoming/Kimi-norm-manifest-r9037.json');inside=change(inside,'/incoming/Kimi_gate_driver_r9032.py','/incoming/Kimi_norm_driver_r9037.py');(b/'GPU_inside.sh').write_bytes(inside.encode());files={'Kimi_norm_driver_r9037.py':b/'norm_driver.py','controller_Kimi_norm_r9037.py':b/'controller_gpu.py','launch_Kimi_norm_r9037.py':b/'launch_gpu.py','Kimi_norm_GPU_r9037_inside.sh':b/'GPU_inside.sh'};m['sources_SHA256']={n:sha(p)for n,p in files.items()};m['WAVE_INSIDE']='Kimi_norm_GPU_r9037_inside.sh';assert "wave='"+m['WAVE_INSIDE']+"'"in(b/'launch_gpu.py').read_text();(b/'GPU-manifest-local.json').write_text(json.dumps(m,indent=2),encoding='utf8');payload=base64.b64encode(zlib.compress(json.dumps({'manifest':m,'files':{n:p.read_text()for n,p in files.items()}}).encode())).decode()
remote="""import pathlib,json,zlib,base64,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';d=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));m=d['manifest']
for n,s in d['files'].items():
 p=i/n;assert not p.exists(),n;assert chr(13)not in s;p.write_bytes(s.encode())
subprocess.run(['bash','-n',str(i/m['WAVE_INSIDE'])],check=True)
for n,h in m['sources_SHA256'].items():assert hashlib.sha256((i/n).read_bytes()).hexdigest()==h,n
p=i/'Kimi-norm-manifest-r9037.json';assert not p.exists();p.write_text(json.dumps(m,indent=2));p=i/'Kimi-norm-rootGO-r9037.json';assert not p.exists();p.write_text(json.dumps({'root_continuous_GO':True,'commit':'private-root-realnorm-GO-r9037'}))
p=subprocess.run(['python3',str(i/'launch_Kimi_norm_r9037.py')],text=True,capture_output=True,timeout=180);assert p.returncode==0,(p.stdout,p.stderr);print(p.stdout)
""".replace('PAYLOAD',repr(payload));compile(remote,'invoke','exec');(r/'invoke_Kimi_norm_GPU_r9037.py').write_bytes(remote.encode());print('CPU266/runtime/source/WAVE/LF contracts PASS')
