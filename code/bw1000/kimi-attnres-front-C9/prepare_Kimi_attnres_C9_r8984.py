import pathlib,json,hashlib,ast,difflib
r=pathlib.Path.cwd();old=r/'Kimi-attnres-front-private-r8982';new=r/'Kimi-attnres-front-C9-private-r8984';new.mkdir(exist_ok=True)
names={'attnres_front_r8982.cu':'attnres_front_C9_r8984.cu','attnres_explicit_r8982.cpp':'attnres_explicit_C9_r8984.cpp','run_attnres_front_r8982.py':'run_attnres_front_C9_r8984.py','cpu_build_plan_r8982.py':'cpu_build_plan_C9_r8984.py'}
for src,dst in names.items():
 text=(old/src).read_text()
 if src.endswith('.cu'):text=text.replace('C = 24, CAP = 25','C = 9, CAP = 9').replace('16 * 24','16 * 9')
 if src.endswith('.cpp'):text=text.replace('blocks != 24 || capacity != 25','blocks != 9 || capacity != 9').replace('M16/C24/CAP25/H7168','M16/C9/CAP9/H7168')
 if src.startswith('run_'):
  for a,b in [('history24/cap25','history9/cap9'),('layer46','layer92'),('layers.46.','layers.92.'),('(16,25,7168)','(16,9,7168)'),('[:,:24]','[:,:9]'),('(16,24,7168)','(16,9,7168)'),('a.device,16,24,25,7168','a.device,16,9,9,7168')]:text=text.replace(a,b)
  text=text.replace("    storage=torch.full((16,9,7168),float('nan'),device='cuda',dtype=torch.bfloat16)","    input_padded=torch.full((16*9*7168+64,),.3125,device='cuda',dtype=torch.bfloat16)\n    storage=input_padded[32:-32].view(16,9,7168)")
  text=text.replace("    assert not storage[:,:9].is_contiguous()  # Actual [CAP*H,H,1] residual-bank stride.","    assert storage.stride()==(9*7168,7168,1)\n    input_reference=storage.clone()")
  text=text.replace("        assert bool(torch.isnan(storage[:,24]).all()), 'STOP_unused_bank_corruption'","        assert torch.equal(storage,input_reference), 'STOP_input_corruption'\n        assert bool((input_padded[:32]==.3125).all() and (input_padded[-32:]==.3125).all()), 'STOP_input_canary'")
  text+="\nif report['status']=='COMPONENT_COMPLETED_NOT_MODEL_QUALIFIED' and report.get('shutdown_returncode')==0:\n    print('KIMI_ATTNRES_FRONT_NORMAL',json.dumps({'normal':True,'counter':report['arms'][-1]['counter']}),flush=True)\n"
 if src.startswith('cpu_'):
  text=text.replace('Kimi-attnres-front-r8982','Kimi-attnres-front-C9-r8984').replace('libattnres_front_r8982.so','libattnres_front_C9_r8984.so')
  for a,b in names.items():text=text.replace(a,b)
 (new/dst).write_bytes(text.encode())
for src,dst in [('build_Kimi_attnres_front_r8983.py','build_Kimi_attnres_C9_r8984.py'),('Kimi_attnres_front_CPU_r8983_inside.sh','Kimi_attnres_C9_CPU_r8984_inside.sh'),('launch_Kimi_attnres_front_CPU_r8983.py','launch_Kimi_attnres_C9_CPU_r8984.py')]:
 text=(r/src).read_text().replace('Kimi-attnres-front-r8982','Kimi-attnres-front-C9-r8984').replace('Kimi-attnres-front-CPU-r8983','Kimi-attnres-C9-CPU-r8984').replace('Kimi-attnres-front-source-r8983.json','Kimi-attnres-C9-source-r8984.json').replace('libattnres_front_r8982.so','libattnres_front_C9_r8984.so').replace('build_Kimi_attnres_front_r8983.py','build_Kimi_attnres_C9_r8984.py').replace('Kimi_attnres_front_CPU_r8983_inside.sh','Kimi_attnres_C9_CPU_r8984_inside.sh')
 for a,b in names.items():text=text.replace(a,b)
 (r/dst).write_bytes(text.encode())
capacity=json.loads((r/'Kimi-attnres-capacity-r8984.json').read_text());allocation=next(x for x in capacity['files']if '/kimi-serving-'in x['path']and x['path'].endswith('kimi_k3_text_model.cpp'))
domain=json.loads((r/'Kimi-attnres-actual-domain-r8984.json').read_text());domain.update(max_persistent_blocks=8,max_values_C=9,formula='ceil(layer/12)+1 plus reset before FFN; layer84..92 max9',fixture_CAP=9,fixture_CAP_note='actual final-stage capacity=ceil(local_layer_end93/12)+1=9; no unused logical slot; input/output padding canaries',allocation_source=allocation['path'],allocation_source_SHA256=allocation['SHA256'],allocation_line45='block_residual_capacity_=(local_layer_end_+block_size-1)/block_size+1')
(new/'actual_domain_r8984.json').write_bytes((json.dumps(domain,indent=2)+'\n').encode())
contract=json.loads((old/'contract_r8982.json').read_text())
contract['scope']='Single-card explicit-weights synthetic component M16/actual maximum operands C9/CAP9/H7168'
contract['fixture']='One seeded synthetic bank, actual layer92 two small weights; two warm + 8 timed ABBA+BAAB + restoreA; input/output padding canaries, no unused logical slot'
contract['stride']='[16,9,7168] actual [64512,7168,1]'
contract['domain_proof']=domain
contract['supersedes']='C24 CPU232 source remains preserved; no C24 GPU was submitted'
(new/'contract_C9_r8984.json').write_bytes((json.dumps(contract,indent=2)+'\n').encode())
diff=''
for a,b in names.items():diff+=''.join(difflib.unified_diff((old/a).read_text().splitlines(True),(new/b).read_text().splitlines(True),fromfile=a,tofile=b))
(new/'C24_to_C9_shape_only.diff').write_bytes(diff.encode())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[x for x in new.iterdir()if x.is_file()]+[r/n for n in ['build_Kimi_attnres_C9_r8984.py','Kimi_attnres_C9_CPU_r8984_inside.sh','launch_Kimi_attnres_C9_CPU_r8984.py','prepare_Kimi_attnres_C9_r8984.py','read_Kimi_attnres_actual_domain_r8984.py']]
source={'status':'ACTUAL_C9_SHAPE_ONLY_CPU_BUILD','source_SHA256':{x.name:sha(x)for x in files if x.suffix in ['.cpp','.cu','.py','.sh']}}
(r/'Kimi-attnres-C9-source-r8984.json').write_bytes((json.dumps(source,indent=2)+'\n').encode())
repo=r/'InfiniLM-hygon-test';dst=repo/'code/bw1000/kimi-attnres-front-C9';dst.mkdir(exist_ok=True)
for x in files+[r/'Kimi-attnres-C9-source-r8984.json']:
 if x.suffix=='.py':ast.parse(x.read_text())
 (dst/x.name).write_bytes(x.read_bytes())
print(json.dumps({'files':len(files)+1,'source_SHA256':sha(r/'Kimi-attnres-C9-source-r8984.json'),'domain':domain}))
