import pathlib,json,hashlib,subprocess,os,time,shutil,struct,re,collections
r=pathlib.Path('/refactor');b=r/'candidates/Kimi-A8-bitfield-r8996';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();started=time.monotonic();d={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'commands':[],'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}
def run(cmd,name):
 d['commands'].append(cmd)
 with(b/(name+'.log')).open('x')as f:z=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,155-(time.monotonic()-started)))
 assert z.returncode==0,(name,(b/(name+'.log')).read_text()[-4000:])
try:
 b.mkdir(exist_ok=False);(b/'src').mkdir();proof=json.loads((r/'incoming/Kimi-A8-bitfield-source-r8996.json').read_text());src=r/'incoming/kimi_A8_bitfield_r8996.hip';assert sha(src)==proof['source_SHA256'][src.name];shutil.copyfile(src,b/'src'/src.name)
 old=json.loads((r/'results/Kimi-A8-ABI-CPU-r8993.json').read_text());assert old['completed']and old['CPU_only']and not old['GPU']and not old['dlopen'];adapter=next(x for x in old['objects']if x.endswith('/adapter.o'));provider=next(iter(old['unchanged_provider_object']));assert sha(adapter)==old['objects'][adapter]and sha(provider)==old['unchanged_provider_object'][provider]
 cmd=list(old['commands'][0]);cmd[cmd.index('-c')+1]=str(b/'src'/src.name);cmd[cmd.index('-o')+1]=str(b/'matrix.o');cmd[cmd.index('-MF')+1]=str(b/'matrix.d');run(cmd,'pureHIP-bitfield')
 native=b/'libKimi_A8_bitfield_r8996.so';cmd=list(old['commands'][2]);cmd[cmd.index('-o')+1]=str(native);cmd=[str(b/'matrix.o')if x.endswith('/matrix.o')else adapter if x.endswith('/adapter.o')else x for x in cmd];run(cmd,'reuse-adapter-provider-link')
 defs=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True);assert all(n in defs for n in ['modern_kimi_A8_W13_r8993','kimi_A8_HIP_matrix_r8993','modern_kimi_grouped_I192_r8669y'])
 raw=(b/'matrix.o').read_bytes();images=[]
 for i in range(len(raw)-64):
  if raw[i:i+4]!=b'\x7fELF'or raw[i+4]!=2 or struct.unpack_from('<H',raw,i+18)[0]!=224:continue
  off=struct.unpack_from('<Q',raw,i+40)[0];ents,num=struct.unpack_from('<HH',raw,i+58)
  if not off or ents!=64 or off+num*64>len(raw)-i:continue
  end=off+num*64
  for k in range(num):
   sh=i+off+k*64;typ=struct.unpack_from('<I',raw,sh+4)[0];so,sz=struct.unpack_from('<QQ',raw,sh+24)
   if typ!=8:end=max(end,so+sz)
  if end>len(raw)-i:continue
  elf=b/f'gpu-{i}.elf';elf.write_bytes(raw[i:i+end]);meta=subprocess.check_output(['/opt/dtk/llvm/bin/llvm-readelf','--notes',str(elf)],text=True,timeout=15);dis=subprocess.check_output(['/opt/dtk/aillvm/bin/llvm-objdump','-d','--mcpu=gfx936',str(elf)],text=True,timeout=15);(b/'matrix.isa.txt').write_text(dis);(b/'matrix.metadata.txt').write_text(meta)
  resource=next(s for s in meta.split('  - .args:')if'w13_a8'in s);assert '.private_segment_fixed_size: 0'in resource and '.sgpr_spill_count: 0'in resource and '.vgpr_spill_count: 0'in resource
  body=dis[dis.index('<_ZN12_GLOBAL__N_16w13_a8'):];counts=collections.Counter(re.findall(r'\b(?:v|s|ds|global|buffer|scratch|flat)_[a-zA-Z0-9_]+',body));assert counts['s_getpc_b64']==0 and counts['global_load_dword']==9 and counts['v_mmac_i32_16x16x32_i8']==1;assert not any('scratch'in x for x in counts)
  images.append({'ELF_SHA256':sha(elf),'metadata':resource,'opcode_counts':dict(counts),'interesting_lines':[s for s in body.splitlines()if any(k in s for k in ['C8643210','c8643210','v_mmac','v_lshrrev_b32','s_barrier'])][:20],'ISA_SHA256':sha(b/'matrix.isa.txt'),'old957_static_LUT_PC_relative':1,'new_static_LUT_PC_relative':counts['s_getpc_b64'],'old957_global_load_dword':10,'new_global_load_dword':counts['global_load_dword']})
 assert len(images)==1
 d.update(completed=True,native=str(native),native_SHA256=sha(native),source_SHA256=proof['source_SHA256'],objects={str(b/'matrix.o'):sha(b/'matrix.o')},reused_objects={adapter:sha(adapter),provider:sha(provider)},defined=defs,needed=subprocess.check_output(['readelf','-d',str(native)],text=True),ISA=images,new_GPU_TUs=1,new_host_TUs=0,newSDK_build=False)
except BaseException as e:d['error']=repr(e);raise
finally:
 d['elapsed_s']=time.monotonic()-started;p=r/'results/Kimi-A8-bitfield-CPU-r8996.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print(json.dumps({k:d.get(k)for k in ['completed','error','step','native','native_SHA256','elapsed_s']}))
