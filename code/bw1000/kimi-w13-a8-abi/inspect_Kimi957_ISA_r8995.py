import pathlib,json,hashlib,struct,subprocess,re,collections
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';obj=r/'candidates/Kimi-W13-A8-ABI-r8993/matrix.o';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert sha(obj)=='159f2498cf88641d93224b80c76142e44688e1a37df3970e579a1b6855240c77'
base=r/'results/Kimi957-ISA-r8995';assert not base.exists();base.mkdir();raw=obj.read_bytes();out={'CPU_only':True,'GPU':False,'dlopen':False,'recompile':False,'object':str(obj),'object_SHA256':sha(obj),'images':[]}
app='/public/software/apps/apptainer/1.3.4/bin/apptainer';cmd=[app,'exec','--bind',str(r)+':/refactor',str(r/'containers/dtk26-sandbox-r1')]
for i in range(len(raw)-64):
 if raw[i:i+4]!=b'\x7fELF'or raw[i+4]!=2 or struct.unpack_from('<H',raw,i+18)[0]!=224:continue
 off=struct.unpack_from('<Q',raw,i+40)[0];ents,num=struct.unpack_from('<HH',raw,i+58)
 if not off or ents!=64 or off+num*64>len(raw)-i:continue
 end=off+num*64
 for k in range(num):
  sh=i+off+k*64;typ=struct.unpack_from('<I',raw,sh+4)[0];so,sz=struct.unpack_from('<QQ',raw,sh+24)
  if typ!=8:end=max(end,so+sz)
 if end>len(raw)-i:continue
 file=base/f'gpu-{i}.elf';file.write_bytes(raw[i:i+end]);cp=str(file).replace(str(r),'/refactor')
 note=subprocess.run(cmd+['/opt/dtk/llvm/bin/llvm-readelf','--notes',cp],capture_output=True,text=True,timeout=25);z=subprocess.run(cmd+['/opt/dtk/aillvm/bin/llvm-objdump','-d','--mcpu=gfx936',cp],capture_output=True,text=True,timeout=25)
 assert note.returncode==0 and z.returncode==0,(note.stderr,z.stderr)
 dis=base/(file.name+'.isa.txt');dis.write_text(z.stdout);meta=base/(file.name+'.metadata.txt');meta.write_text(note.stdout)
 chunks=re.split(r'(?m)^([0-9a-f]+ <[^>]+>:)\s*$',z.stdout);kernels=[]
 for j in range(1,len(chunks),2):
  header=chunks[j];body=chunks[j+1]
  if'w13_a8'not in header and'code_i8'not in header:continue
  lines=body.splitlines();op=collections.Counter(re.findall(r'\b(?:v|s|ds|global|buffer|scratch|flat)_[a-zA-Z0-9_]+',body));kernels.append({'header':header,'opcode_counts':dict(op),'interesting_lines':[x for x in lines if any(k in x.lower()for k in ['mma','barrier','scratch','private','flat_load','buffer_load'])][:48],'line_count':len(lines)})
 out['images'].append({'ELF_SHA256':sha(file),'metadata':note.stdout[:24000],'ISA_path':str(dis),'ISA_SHA256':sha(dis),'ISA_bytes':dis.stat().st_size,'kernels':kernels,'stderr':z.stderr[-1000:]})
assert out['images'];print(json.dumps(out))
