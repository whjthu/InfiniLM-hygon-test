import pathlib,json,hashlib,subprocess,time,os,re
r=pathlib.Path('/refactor');i=r/'incoming';b=r/'candidates/glm-gate-fragment-r9048';b.mkdir();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();q={'completed':False,'CPU_only':True,'GPU':False,'dlopen':False,'GPU_TUs_compiled_on_CPU':1,'HOST_TUs_compiled':0,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'commands':[]};start=time.perf_counter()
def run(a):
 q['commands'].append(a);p=subprocess.run(a,text=True,capture_output=True,timeout=110);(b/f'command-{len(q["commands"])}.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr[:16000];return p.stdout
def instructions(text,symbol):
 pos=text.index(symbol+':');end=re.search(r'(?m)^\.Lfunc_end\d+:',text[pos:]);assert end,('functionend',symbol);s=text[pos:pos+end.end()];rows=[]
 for line in s.splitlines():
  if re.match(r'\s+(?:v_|s_|global_|ds_|flat_|buffer_|scratch_)[a-z0-9_]+\s?',line):
   v=line.strip().split('//')[0].strip();v=re.sub(r'\.LBB\d+_',r'.LBB_',v);v=re.sub(r'\.LCPI\d+_',r'.LCPI_',v);rows.append(v)
 assert rows;return rows
try:
 parent=json.loads((r/'candidates/glm-reduced-output-r9034/CPU-build.json').read_bytes());assert parent['completed'];link=next(list(a)for a in parent['commands']if'-shared'in a);objects={x:sha(x)for x in link if x.endswith('.o')};assert len(objects)==28;old='/refactor/candidates/glm-vector-r8529/vector.o';assert objects[old]=='75f6ecd450268db5b0f062c2859e1a9f8f8bd2814730afccbc17f9486bd4fd80'
 src=i/'glm_gate_fragment_r9048.cu';oldsrc=i/'glm_scalar_vectorload_r8528.cu';source=src.read_text();original=oldsrc.read_text();a=original.index('__global__ void gate_up(');z=original.index('__global__ void down(',a);assert original[a:z]in source
 nvcc='/opt/dtk/cuda/cuda/bin/nvcc';asm=b/'fragment.s';obj=b/'fragment.o';run([nvcc,'-std=c++17','-O3','-arch=gfx936','-S',str(src),'-o',str(asm)]);run([nvcc,'-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-MD','-MF',str(b/'fragment.d'),'-c',str(src),'-o',str(obj)])
 text=asm.read_text();previous=(r/'candidates/glm-vector-r8529/vector.s').read_text();gate='_ZN12_GLOBAL__N_17gate_upEP13__nv_bfloat16S1_PKhPKfPKlS3_S5_S3_S5_mm';down='_ZN12_GLOBAL__N_14downEP13__nv_bfloat16PKhPKfPKlS3_S5_mm';bind={}
 for name,symbol in[('gate_original',gate),('down_unchanged',down)]:
  x=instructions(previous,symbol);y=instructions(text,symbol);bind[name]={'canonical_instruction_equal':x==y,'old_n':len(x),'new_n':len(y),'old_SHA256':hashlib.sha256('\n'.join(x).encode()).hexdigest(),'new_SHA256':hashlib.sha256('\n'.join(y).encode()).hexdigest()};assert x==y,('baseline_ISA_changed',name)
 metadata=[]
 for block in re.split(r'(?m)^  - \.args:',text):
  name=re.search(r'(?m)^\s+\.name:\s+(\S+)',block)
  if name:
   v={'name':name.group(1)}
   for key in['vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size']:
    match=re.search(r'(?m)^\s+\.'+key+r':\s+(\d+)',block);v[key]=int(match.group(1))if match else None
   metadata.append(v)
 assert len(metadata)>=4,('ISAmetadata',metadata);assert 'v_mmac_f32_16x16x16_bf16' in text, 'missing_actual_BF16MMAC_ISA';q['actual_BF16MMAC_instruction_lines']=[v.strip()for v in text.splitlines()if 'v_mmac_f32_16x16x16_bf16'in v][:12]
 lib=b/'libglm_gate_fragment_r9048.so';link=[str(obj)if x==old else x for x in link];link[link.index('-o')+1]=str(lib);run(link);nm=run(['nm','-D','--defined-only',str(lib)]);assert all(s in nm for s in['glm_gate_fragment_policy_r9048','glm_gate_fragment_counts_r9048','glm_vec_gate_r8528','glm_vec_down_r8528','glm_layer_forward_r5670']);run(['readelf','-d',str(lib)]);assert all(sha(p)==h for p,h in objects.items())
 q.update(completed=True,library=str(lib),library_SHA256=sha(lib),source_SHA256=sha(src),object_SHA256=sha(obj),ISA_SHA256=sha(asm),original_ISA_bind=bind,ISA_metadata=metadata,reused_objects_SHA256={p:h for p,h in objects.items()if p!=old},elapsed_seconds=time.perf_counter()-start,scope='ONEGPU TU -S/-c same -O3 gfx936 flags; baselinegate/down canonicalISAequal;27objects/HOST/SDK unchanged; fragmentMMACdefaultOFF')
except BaseException as e:q['failure']=repr(e)
(b/'CPU-build.json').write_text(json.dumps(q,indent=2));print(json.dumps(q));assert q['completed']
