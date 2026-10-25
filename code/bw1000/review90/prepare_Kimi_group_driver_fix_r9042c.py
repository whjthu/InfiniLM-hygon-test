import pathlib,json,hashlib,ast,base64,zlib,re
old=pathlib.Path('Kimi-rowreduce-group-private-r9042');b=pathlib.Path('Kimi-rowgroup-active-private-r9042c');b.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();source=(old/'delta-final.cpp').read_text();envs=re.findall(r'flag9042\("([A-Z0-9_]+)"\)',source);assert envs==['INFINILM_KIMI_ROWGROUP_R9042','INFINILM_KIMI_ROWGROUP_TRACE_R9042'];points=re.findall(r'trace_kda_projection\(runtime,output,"([^"]+)"\)',source);assert points==['r9042_o_proj_local','r9042_o_proj_reduced']
s=(old/'layer0_driver.py').read_text().replace('Kimi-layer0-real-r9042-','Kimi-layer0-real-r9042c-').replace('Kimi-layer0-raw-r9042-','Kimi-layer0-raw-r9042c-');s=s.replace('INFINILM_KIMI_OPROJ_W1_R9040','INFINILM_KIMI_ROWGROUP_R9042').replace('INFINILM_KIMI_OPROJ_TRACE_R9040','INFINILM_KIMI_ROWGROUP_TRACE_R9042');assert 'OPROJ_'not in s
helper='''GROUP_ENV='INFINILM_KIMI_ROWGROUP_R9042'
TRACE_ENV='INFINILM_KIMI_ROWGROUP_TRACE_R9042'
def apply_flags(candidate,width,trace):
 values={GROUP_ENV:'1'if candidate and width==8 else'0',TRACE_ENV:'1'if trace else'0'};os.environ.update(values);return values
def control_contract():
 rows=[]
 for arm in ['A','B','C']:
  for width in [512,1,8]:
   got=apply_flags(arm=='C',width,True);expected=arm=='C'and width==8;assert(got[GROUP_ENV]=='1')==expected;rows.append({'arm':arm,'width':width,'set':dict(got)})
 apply_flags(False,1,False)
 return {'status':'ACTUAL_CONTROL_FUNCTION_CPU_PASS','consumed_set_ENV':[GROUP_ENV,TRACE_ENV],'rows':rows,'prefix_OFF_final_C_W8_ON':True}
'''
assert s.count('def frontend(manifest):')==1;s=s.replace('def frontend(manifest):',helper+'def frontend(manifest):');s=s.replace("m=json.loads(pathlib.Path(manifest).read_text());","m=json.loads(pathlib.Path(manifest).read_text());controlproof=control_contract();assert m['consumed_ENV']==[GROUP_ENV,TRACE_ENV];assert m['required_points']==['r9042_o_proj_local','r9042_o_proj_reduced'];")
s=s.replace("'planned_native_calls_per_rank':345,'planned_native_calls_allranks':5520","'planned_native_calls_per_rank':371,'planned_native_calls_allranks':5936,'control_contract':controlproof")
s=s.replace("for arm in m['arms']:(root/arm).mkdir()","for arm in ['W',*m['arms']]:(root/arm).mkdir()")
s=s.replace("os.environ['INFINILM_KIMI_ROWGROUP_R9042']='0';os.environ['INFINILM_KIMI_ROWGROUP_TRACE_R9042']='1';","apply_flags(False,1,True);")
s=s.replace("os.environ['INFINILM_KIMI_ROWGROUP_R9042']='1'if arm=='C'and width==8 else'0'","apply_flags(arm=='C',width,True)")
s=s.replace("os.environ['INFINILM_KIMI_ROWGROUP_TRACE_R9042']='0';os.environ['INFINILM_KIMI_ROWGROUP_R9042']='0'","apply_flags(False,1,False)")
s=s.replace("os.environ['INFINILM_KIMI_ROWGROUP_R9042']='1'if candidate and width==8 else'0'","apply_flags(candidate,width,False)")
needle="   for arm,widths in m['arms'].items():"
warm='''   # Before the full matrix, prove selected branch execution with trace-control OFF.
   if device==0:os.environ['INFINILM_KIMI_TRACE_DIR']=str(root/'W');control.collect('activation_warm_begin',{'node':node});report['phase']='activation_warm';save()
   barrier.wait();check(lib.layer0_reset_r9038(ctx));active,shadow=1,9;past=0;own['activation_warm_calls']=[]
   for width in [512,*([1]*24),8]:
    if device==0:apply_flags(True,width,False)
    barrier.wait();read=1 if past==0 else active;write=1 if past==0 else shadow;x=np.ascontiguousarray(norm[past:past+width]);guard=np.full(width*7168+64,0x7fc1,np.uint16);out=guard[32:-32].reshape(width,7168);before=hashlib.sha256(x.tobytes()).hexdigest();check(lib.layer0_run_r9038(ctx,width,read,write,x.ctypes.data,out.ctypes.data));barrier.wait();assert np.all(guard[:32]==0x7fc1)and np.all(guard[-32:]==0x7fc1);assert not np.any((out&0x7f80)==0x7f80);assert hashlib.sha256(x.tobytes()).hexdigest()==before;own['activation_warm_calls'].append({'past':past,'width':width,'read':read,'write':write,'output_SHA256':hashlib.sha256(out.tobytes()).hexdigest()})
    if past>0:active,shadow=shadow,active
    past+=width
   assert past==544;barrier.wait()
   if device==0:
    witness=[]
    for localdev in range(8):
     p=root/'W'/f'stage{node}-rank{localdev}-layer0-kda-call25.jsonl';rows=[json.loads(z)for z in p.read_text().splitlines()];found=[z for z in rows if z['point']in m['required_points']];assert len(found)==2 and{z['point']for z in found}==set(m['required_points']);assert all(z['shape']==[1,8,7168]and z['nonfinite']==0 for z in found);witness.append({'rank':node*8+localdev,'journal':str(p),'journal_SHA256':sha(p),'points':[z['point']for z in found]})
    proof={'node':node,'passed':True,'candidate_group_ENV':os.environ[GROUP_ENV],'trace_control_ENV':os.environ[TRACE_ENV],'witness':witness,'completed_sections':8,'physical_collectives_per_section_from_pinned_branch':8,'independent_native_counter':False};assert proof['candidate_group_ENV']=='1'and proof['trace_control_ENV']=='0';report['activation_witness']=proof;save();allproof=control.collect('all16_activation_witness',proof);assert len(allproof)==2 and all(z['passed']for z in allproof)
   barrier.wait()
'''
assert s.count(needle)==1;s=s.replace(needle,warm+needle)
s=s.replace("for z in report['workers']),Model_finalIDs", "+len(z['activation_warm_calls'])for z in report['workers']),Model_finalIDs")
s=s.replace('==2760','==2968').replace("'native_calls':2760","'native_calls':2968")
ast.parse(s);(b/'layer0_driver.py').write_bytes(s.encode())
for name in ['controller_gpu.py','launch_gpu.py','GPU_inside.sh','CPU_inside.sh']:
 t=(old/name).read_text();t=t.replace('Kimi-layer0-real-r9042','Kimi-layer0-real-r9042c').replace('Kimi-layer0-manifest-r9042.json','Kimi-layer0-manifest-r9042c.json').replace('Kimi-layer0-ownerfront-r9042','Kimi-layer0-ownerfront-r9042c').replace('Kimi-layer0-post-r9042','Kimi-layer0-post-r9042c').replace('Kimi-layer0-idle-r9042','Kimi-layer0-idle-r9042c').replace('Kimi_layer0_GPU_r9042_inside.sh','Kimi_layer0_GPU_r9042c_inside.sh').replace('Kimi_layer0_driver_r9042.py','Kimi_layer0_driver_r9042c.py').replace('controller_Kimi_layer0_r9042.py','controller_Kimi_layer0_r9042c.py').replace('GLOBAL_RUN_R2354=layer0-r9042','GLOBAL_RUN_R2354=layer0-r9042c').replace('==2760','==2968')
 if name.endswith('.py'):ast.parse(t)
 (b/name).write_bytes(t.encode())
# Execute the actual pure control function and verify source-consumed ENV/parser inventory before upload.
module=ast.parse(s);subset=ast.Module(body=[x for x in module.body if isinstance(x,(ast.Assign,ast.FunctionDef))and((isinstance(x,ast.FunctionDef)and x.name in ['apply_flags','control_contract'])or(isinstance(x,ast.Assign)and any(isinstance(y,ast.Name)and y.id in ['GROUP_ENV','TRACE_ENV']for y in x.targets)))],type_ignores=[]);namespace={'os':__import__('os')};exec(compile(subset,'actual-driver-controls','exec'),namespace);proof=namespace['control_contract']();assert proof['consumed_set_ENV']==envs;(b/'local-control-contract.json').write_text(json.dumps({'actual_function':proof,'required_source_points':points,'driver_SHA256':sha(b/'layer0_driver.py'),'actual_DSO_source_SHA256':sha(old/'delta-final.cpp'),'no_CXX_recompile':True},indent=2))
files={'Kimi_layer0_driver_r9042c.py':b/'layer0_driver.py','controller_Kimi_layer0_r9042c.py':b/'controller_gpu.py','launch_Kimi_layer0_r9042c.py':b/'launch_gpu.py','Kimi_layer0_GPU_r9042c_inside.sh':b/'GPU_inside.sh','Kimi_layer0_owner_CPU_r9042c_inside.sh':b/'CPU_inside.sh'};extra={'sources_SHA256':{n:sha(p)for n,p in files.items()},'WAVE_INSIDE':'Kimi_layer0_GPU_r9042c_inside.sh','consumed_ENV':envs,'required_points':points,'planned_native_calls_per_rank':371,'planned_native_calls_allranks':5936,'repair_of':'910375.693_INERT_SELECTOR_0validrounds','first_candidate_warm_before_matrix':True,'local_control_contract_SHA256':sha(b/'local-control-contract.json')};payload=base64.b64encode(zlib.compress(json.dumps({'files':{n:p.read_text()for n,p in files.items()},'extra':extra}).encode())).decode()
remote="""import pathlib,json,zlib,base64,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');i=r/'incoming';d=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)));sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((i/'Kimi-layer0-manifest-r9042.json').read_text());m.update(d['extra'])
for n,s in d['files'].items():
 p=i/n;assert not p.exists(),n;assert chr(13)not in s;p.write_bytes(s.encode());assert sha(p)==m['sources_SHA256'][n]
for n in ['Kimi_layer0_GPU_r9042c_inside.sh','Kimi_layer0_owner_CPU_r9042c_inside.sh']:subprocess.run(['bash','-n',str(i/n)],check=True)
p=i/'Kimi-layer0-manifest-r9042c.json';assert not p.exists();p.write_text(json.dumps(m,indent=2));args=['srun','--jobid=910375','--overlap','--exact','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w','a02r4n08,a02r4n09','--gres=none','--mem=8192M','-t','00:03:00','--job-name=Kimi-layer0-ownerfront-r9042c','--output='+str(r/'logs/Kimi-layer0-ownerfront-r9042c-%J-%t.log'),'env','WAVE_INSIDE_R2323=Kimi_layer0_owner_CPU_r9042c_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/Kimi-layer0-ownerfront-r9042c-launch.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_only':True,'GPU':False,'pid':p.pid,'args':args,'manifest_SHA256':sha(i/'Kimi-layer0-manifest-r9042c.json')}))
""".replace('PAYLOAD',repr(payload));compile(remote,'upload','exec');pathlib.Path('invoke_Kimi_group_front_r9042c.py').write_bytes(remote.encode());print('actualcontrols/sourceENV/pointinventory pass;371calls/rank')
