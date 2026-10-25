import pathlib,ast,json
r=pathlib.Path.cwd();s=(r/'Deep_latent_mapping_component_r8946.py').read_text();s=s.replace('deep-latent-mapping-r8946','deep-reference-scratch-r8954').replace('Deep-latent-mapping-manifest-r8946','Deep-reference-scratch-manifest-r8954');s=s.replace("lib.deep_latent_prefill_counts_r8933.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]", "lib.deep_reference_scratch_mode_r8953.argtypes=[ctypes.c_int];lib.deep_reference_scratch_mode_r8953.restype=ctypes.c_int;lib.deep_reference_scratch_counts_r8954.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];lib.deep_reference_scratch_counts_r8954.restype=ctypes.c_int")
s=s.replace("level='SYNTHETIC_ATTENTION_MAPPING_COMPONENT'", "level='REFERENCE_ATTENTION_SCRATCH_COMPONENT'")
s=s.replace('q=(ctypes.c_uint64*3)();check(lib.deep_latent_prefill_counts_r8933(d,q,3));return list(q)','q=(ctypes.c_uint64*4)();check(lib.deep_reference_scratch_counts_r8954(d,q,4));return list(q)')
s=s.replace('def invoke(d,which,width):','def invoke(d,which):\n width=2048;arm=[0,1,1,0][which]')
s=s.replace("assert delta==([1,2048,64]if which==1 else[0,0,0]);", "assert delta==([0,1,0,3]if arm else[1,0,6144,0]);")
s=s.replace("report['CPU_map_contract']=validate();save()", "report['CPU_map_contract']=validate();check(lib.deep_latent_prefill_mode_r8933(0));check(lib.deep_reference_scratch_mode_r8953(0));assert lib.deep_reference_scratch_mode_r8953(2)==1;save()")
a=s.index('  for which,width,mode in ');b=s.index('  together(lambda d:state.pop(d));',a)
s=s[:a]+'''  for which,arm in enumerate([0,1,1,0]):
   check(lib.deep_reference_scratch_mode_r8953(arm));row['measurements'].append(dict(output=which,arm=arm,owners=together(invoke,which)));save()
  row['comparisons']={str((a,b)):together(compare,a,b)for a,b in [(0,1),(0,3),(1,2)]}
  row['ownrepeat_A']=all(v['maxabs']==0 for v in row['comparisons']['(0, 3)']);row['ownrepeat_B']=all(v['maxabs']==0 for v in row['comparisons']['(1, 2)']);assert row['ownrepeat_A']and row['ownrepeat_B'];save()
'''+s[b:]
s=s.replace('COMPLETED_DIAGNOSTIC_NO_MODEL_QUALIFICATION','COMPLETED_REFERENCE_SCRATCH_COMPONENT_ONLY').replace('ThreefixedB4I512 source-derivedglobalmaps, independentlocalRNG per8owners;syntheticRMS+standardRoPE notactualactivations/Yarn/learnedcompressedKV. NoModelweightsCCL/F64. Numericnonbitexactdiagnostic,Modeltokenruleunchanged.','ReferenceFP32Ops scratchOFF/ON ABBA,3fixedB4I512maps/8independentownerRNG;original8933fusedOFF,allT2048callstakereference. SyntheticRMS/RoPEonly,noModelweightsCCL/F64. SameGemmSoftmaxCastargumentsshapesstridesorderinganddrain. Crossarmerrorstatsdiagnostic,ownrepeatrequired;Modeltokenruleunchanged.')
ast.parse(s);(r/'Deep_reference_scratch_component_r8954.py').write_bytes(s.encode())
def rename(s):return s.replace('Deep_latent_mapping_component_r8946','Deep_reference_scratch_component_r8954').replace('deep-latent-mapping-r8946','deep-reference-scratch-r8954').replace('Deep-latent-mapping-manifest-r8946','Deep-reference-scratch-manifest-r8954')
for src,dst in [('Deep_latent_mapping_component_r8946_inside.sh','Deep_reference_scratch_component_r8954_inside.sh'),('launch_Deep_latent_mapping_component_r8946.py','launch_Deep_reference_scratch_component_r8954.py')]:
 ss=rename((r/src).read_text());(r/dst).write_bytes(ss.encode());dst.endswith('.py')and ast.parse(ss)
print('component source prepared;CPU receipt/DSO pending;noGPU')
