import pathlib
s=pathlib.Path('prepare_Kimi_oproj_GPU_r9040.py').read_text().replace('r9040','r9042').replace('Kimi-oproj-canonical','Kimi-rowreduce-group').replace('INFINILM_KIMI_OPROJ_W1_R9042','INFINILM_KIMI_ROWGROUP_R9042').replace('INFINILM_KIMI_OPROJ_TRACE_R9042','INFINILM_KIMI_ROWGROUP_TRACE_R9042').replace('OPROJ_CANONICAL_COMPONENT_DIAGNOSTIC_COMPLETE','ROWGROUP_COMPONENT_DIAGNOSTIC_COMPLETE').replace('invoke_Kimi_oproj_front_r9042.py','invoke_Kimi_group_front_r9042.py').replace('CPU-observed-a.json','CPU-observed-final.json').replace("CPU_build='910407.277'","CPU_build=receipt['producer']")
needle="ast.parse(s);(b/'layer0_driver.py').write_bytes(s.encode())"
rep="""needle=\" def check(code):\"
groupcheck=''' report['RCCL_group_bindings']={}
 maininfo=Info();delta=next(k for k in m['provider_relocation_slots']if 'KimiK3DeltaAttention'in k);p=ctypes.c_void_p.from_address(native_base+m['provider_relocation_slots'][delta]).value;assert dl.dladdr(ctypes.c_void_p(p),ctypes.byref(maininfo));assert sha(maininfo.file.decode())==m['actual678_DSO_SHA256']
 for name,offset in m['RCCL']['Main_GOT_group_slots'].items():
  address=ctypes.c_void_p.from_address(maininfo.base+offset).value;inf=Info();assert address and dl.dladdr(ctypes.c_void_p(address),ctypes.byref(inf));assert sha(inf.file.decode())==m['RCCL']['SHA256'];report['RCCL_group_bindings'][name]={'path':inf.file.decode(),'SHA256':sha(inf.file.decode()),'method':'actualMainGOT','offset':offset}
'''
assert s.count(needle)==1;s=s.replace(needle,groupcheck+needle)
ast.parse(s);(b/'layer0_driver.py').write_bytes(s.encode())"""
assert s.count(needle)==1;s=s.replace(needle,rep)
s=s.replace("extra={k:receipt[k]for k in fields};extra.update(","extra={k:receipt[k]for k in fields};extra['RCCL']=receipt['RCCL'];extra.update(")
compile(s,'generated-group-prepare','exec');pathlib.Path('prepare_Kimi_group_GPU_generated_r9042.py').write_bytes(s.encode());exec(compile(s,'generated-group-prepare','exec'))
