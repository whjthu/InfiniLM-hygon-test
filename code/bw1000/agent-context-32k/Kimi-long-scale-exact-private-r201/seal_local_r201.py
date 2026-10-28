import ast,hashlib,json,pathlib
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in D.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
assert all(b'\r'not in p.read_bytes()for p in D.glob('*.sh'))
m={'schema':1,'scope':'SCOPED_LONG179_PERSISTENT_SCALE_EXACT_CPU_CANDIDATE','GPU_GO':False,'Model_qualified':False,'source_SHA256':{p.relative_to(D).as_posix():sha(p)for p in sorted(D.iterdir())if p.is_file()and p.name not in ['source-manifest-r201.json','CPU-local-contract-r201.json','CPU-contract-r201.exe']},'CPU_entry':'CPU_build_inside_r201.sh','new_HOST_TUs':1,'new_HIP_TUs':0,'old_defaults_unchanged':True,'context':32768,'logical_physical_pool':[1,2,3],'ordinary_thinking_output_budgets':[4096,8192],'4GiB_floor_bytes':4294967296,'actual_device_saving':'UNKNOWN','full32K_peak':'UNKNOWN','actual_model_quality':'NOT_RUN'}
p=D/'source-manifest-r201.json';p.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'CPU_only':True,'GPU':False,'manifest_SHA256':sha(p)}))
