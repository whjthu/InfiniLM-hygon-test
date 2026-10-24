# Source-only generator correction. Retarget namespace before future use; do not rerun against historical8838.
import pathlib,json,hashlib,ast
r=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=(r/'GLM_tile2_component_r8836.py').read_text().replace('tile2','quant_pair').replace('tile2','quant_pair').replace('8836','8838').replace('glm-tile2-r8831','glm-quant-pair-r8835').replace('0e8a8354e8f6db9d672f4245534736cb5e3c2b9ae8e42371f372e0d26f87a0db','7afe5d194c88bdff75bdd7a4a7905f9313aa1c6db7c745ea5afc74b1bc9d8da2')
# Use hyphenated package/result names consistently; Python module uses underscores.
s=s.replace('GLM-quant_pair','GLM-quant-pair').replace("base=r/'candidates/glm-quant_pair-r8831'","base=r/'candidates/glm-quant-pair-r8835'")
s=s.replace('modern_glm_quant_pair_policy_r8830','modern_glm_quant_reuse_policy_r8832').replace('modern_glm_quant_pair_hits_r8830','modern_glm_quant_reuse_hits_r8832')
s=s.replace(' for name in ["glm_vec_gate_r8528","glm_quant_pair_gate_r8830"]:getattr(lib,name).argtypes=[V]*9+[Z,Z,V]\n','')
s=s.replace("len(proof['compiled'])==2 and len(proof['reused_objects_SHA256'])==26","len(proof['compiled'])==1 and len(proof['reused_objects_SHA256'])==27")
s=s.replace('loadedbytes=0;projection_cpu={};started=time.monotonic()','loadedbytes=0;started=time.monotonic()')
a=s.index('     matched=re.fullmatch(');b=s.index("     desc.append(",a);s=s[:a]+s[b:]
a=s.index('  # Projection and enclosing-layer fixtures');b=s.index('  generator=torch.Generator',a);s=s[:a]+s[b:]
a=s.index("  q['fixture_provenance']=");b=s.index("  reference=None;q['trials']=[]",a)
s=s[:a]+"  q['fixture_provenance']={'producer':'original8707 seed8703 BF16 prefix[4096,6144] + separate M8 hidden[8,6144]','past':512,'count':1,'current_Model_decode_capture':False,'weight_checkpoint_read_passes':1,'projection_quant_bytes_scales_direct_check':'not exposed by existing layerAPI; full layer output comparison is numerical acceptance'}\n"+s[b:]
s=s.replace('quant_pair_hits','quant_reuse_hits').replace("'candidate':'quant_pair gate-only routes64/defaultOFF; vectorON,downoriginal; prefixON fixed'","'candidate':'prepare same-input qa/ka quant reuse defaultOFF; vectorON/downoriginal/prefixON fixed; isolated candidate only'")
p=r/'GLM_quant_pair_component_r8838.py';ast.parse(s);p.write_bytes(s.encode())
for name in ['controller_GLM_tile2_component_r8836.py','launch_GLM_tile2_component_r8836.py','GLM_tile2_component_r8836_inside.sh']:
 s=(r/name).read_text().replace('tile2','quant_pair').replace('8836','8838').replace('GLM-quant_pair','GLM-quant-pair')
 if name.startswith('controller'):
  s=s.replace('quant_pair_hits','quant_reuse_hits')
  s=s.replace(" and len(d['projection_trials'])==11",'');a=s.index(";assert all(a['full_output_bits_exact']and a['canary']");b=s.index(";assert not d['fixture_provenance']",a);s=s[:a]+s[b:]
 if name.startswith('launch'):
  s=s.replace("for node in ['a02r4n12','a02r4n13']:","for node in ['a02r4n13']:").replace("assert 'a02r4n12'in nodes","assert 'a02r4n13'in nodes")
 dst=r/name.replace('tile2','quant_pair').replace('8836','8838');dst.write_bytes(s.encode());ast.parse(s)if dst.suffix=='.py'else None
m=json.loads((r/'GLM-tile2-component-manifest-r8836.json').read_text());old='candidates/glm-tile2-r8831/libglm_tile2_r8831.so';new='candidates/glm-quant-pair-r8835/libglm_quant_pair_r8835.so';m['DSO_SHA256'].pop(old);m['DSO_SHA256'][new]='7afe5d194c88bdff75bdd7a4a7905f9313aa1c6db7c745ea5afc74b1bc9d8da2';m.update(candidate_DSO_SHA256=m['DSO_SHA256'][new],CPU_handle='910407.193',CPU_receipt_SHA256=sha(r/'GLM-quant-pair-CPU-observed-r8835-a.json'),scope='Independent attention same-input qa/ka quant reuse only; current vectorON/tile2 absent/downoriginal; seed8703 syntheticM8 wholelayer38 past512/count1; no realModelcapture/TPS',new_CPU_scope='1actual preparehostTU;27objects reused; originalMD header79b5 unchanged;defaultOFF',runner_actual_DSO_paths=[new]);m['matrix']['candidate']='quant reuse0/1; tile2 absent';m['matrix'].pop('projection',None);m['resource']['one_of_nodes']=['a02r4n13'];m['mechanical_prior']='CPU191 wrongquotedheader retained;CPU193 preciseMD header repair PASS; GPU543 basename failure unrelated preserved';m['sources_SHA256']={n:sha(r/n)for n in ['GLM_quant_pair_component_r8838.py','GLM_quant_pair_component_r8838_inside.sh','controller_GLM_quant_pair_component_r8838.py','launch_GLM_quant_pair_component_r8838.py','resource_wave_local_r3097_outer.sh','verify_gpu_idle_r5819.py']};m['metrics']=['wholelayer_native_wall_ms','wholelayer_driver_CAPI_ms','owner_threadCPU_ms','quant_reuse_hits'];p=r/'GLM-quant-pair-component-manifest-r8838.json';p.write_bytes((json.dumps(m,indent=2)+'\n').encode())
s=(r/'dispatch_GLM_tile2_component_r8836.py').read_text().replace('tile2','quant_pair').replace('GLM-quant_pair','GLM-quant-pair').replace('8836','8838').replace('d34dbb3381331ba9d6357118bfdf2745e67b3d090448f9837cc1e8f142a16560',sha(p));e=r/'dispatch_GLM_quant_pair_component_r8838.py';ast.parse(s);e.write_bytes(s.encode());print('manifest',sha(p),'entry',sha(e))
# Literal audit is deliberately local and specific to this package.
for n in ['GLM_quant_pair_component_r8838.py','controller_GLM_quant_pair_component_r8838.py','launch_GLM_quant_pair_component_r8838.py']:
 text=(r/n).read_text();assert 'tile2'not in text and '8836'not in text and 'prefill-prefix'not in text,(n,'old path');print(n,'AST/literal PASS')
