"""Bounded ELF/ISA consumer, reusing r8996's actual DTK object inspection path.

No device/Runtime/Native execution. Descriptor offsets/bits follow LLVM AMDHSA
KernelDescriptor.h; actual raw descriptors and tool notes are preserved.
"""
from pathlib import Path
import hashlib,re,struct
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sections(raw):
 assert raw[:6]==b'\x7fELF\x02\x01'and len(raw)>=64
 base=struct.unpack_from('<Q',raw,40)[0];width,count=struct.unpack_from('<HH',raw,58)
 assert width==64 and 0<count<4096 and base+width*count<=len(raw)
 return [struct.unpack_from('<IIQQQQIIQQ',raw,base+i*64)for i in range(count)]

def descriptors(raw):
 ss=sections(raw);found={}
 for s in ss:
  if s[1]not in [2,11]:continue # SYMTAB/DYNSYM
  assert s[9]==24 and s[6]<len(ss)and s[4]+s[5]<=len(raw)
  strings=ss[s[6]];table=raw[strings[4]:strings[4]+strings[5]]
  for off in range(s[4],s[4]+s[5],24):
   name,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,off)
   assert name<len(table);end=table.find(b'\0',name);assert end>=0
   key=table[name:end].decode(errors='strict')
   if not key.endswith('.kd')or not any(t in key for t in ['gemv_kernel','cast_kernel']):continue
   assert 0<index<len(ss);section=ss[index];local=value-section[3]
   assert 0<=local and local+64<=section[5]and size in [0,64]
   at=section[4]+local;descriptor=raw[at:at+64];rsrc1=struct.unpack_from('<I',descriptor,48)[0]
   data=dict(symbol=key,descriptor_file_offset=at,descriptor_hex=descriptor.hex(),compute_pgm_rsrc1=rsrc1,F32_round_mode=(rsrc1>>12)&3,F32_denorm_mode=(rsrc1>>16)&3)
   assert data['F32_round_mode']==0 and data['F32_denorm_mode']==3,data
   if key in found:assert found[key]==data
   found[key]=data
 assert len(found)==3 and sum('gemv_kernel'in n for n in found)==2,found
 return list(found.values())

def inspect(obj,out,run):
 obj,out=Path(obj),Path(out);raw=obj.read_bytes();images=[];cursor=0
 while True:
  offset=raw.find(b'\x7fELF',cursor)
  if offset<0:break
  cursor=offset+4
  if offset+64>len(raw)or raw[offset+4:offset+6]!=b'\x02\x01'or struct.unpack_from('<H',raw,offset+18)[0]!=224:continue
  start=raw[offset:];ss=sections(start);end=struct.unpack_from('<Q',start,40)[0]+64*len(ss)
  for section in ss:
   if section[1]!=8:end=max(end,section[4]+section[5])
  assert end<=len(start);image=out/('GPU-image-'+str(len(images))+'.elf');image.write_bytes(start[:end]);cursor=offset+end
  notes=run(['/opt/dtk/llvm/bin/llvm-readelf','--notes',str(image)],'GPU'+str(len(images))+'.notes')
  dis=run(['/opt/dtk/aillvm/bin/llvm-objdump','-d','--mcpu=gfx936',str(image)],'GPU'+str(len(images))+'.ISA')
  assert not re.search(r'\bv_(?:mfma|mmac)[_a-z0-9]*',dis,re.I),'Matrix opcode forbidden in strict scalar leaves'
  chunks=re.split(r'(?m)^[0-9a-f]+ <([^\n]+)>:\s*\n',dis);kernels=[]
  for j in range(1,len(chunks),2):
   name,body=chunks[j:j+2]
   if 'gemv_kernel'not in name:continue
   fma=re.findall(r'\bv_fma(?:c)?_f32(?:_e(?:32|64))?\b',body)
   assert fma,dict(kernel=name,reason='No scalar F32 FMA opcode')
   assert not re.search(r'\bv_[a-z0-9_]*(?:f16|bf16|f64)\b',body),name
   kernels.append(dict(symbol=name,scalar_F32_FMA_count=len(fma)))
  assert len(kernels)==2,kernels
  images.append(dict(path=str(image),SHA256=sha(image),bytes=end,ELF_flags=struct.unpack_from('<I',start,48)[0],target_requested='gfx936',descriptors=descriptors(start[:end]),kernels=kernels,notes_log_SHA256=sha(out/('GPU'+str(len(images))+'.notes.log')),ISA_log_SHA256=sha(out/('GPU'+str(len(images))+'.ISA.log')),scope='Static scalarFMA/descriptorIEEE evidence, not actual device numerical proof'))
 assert len(images)==1,images
 return dict(passed=True,object_SHA256=sha(obj),images=images,GPU=False,Runtime=False,descriptor_reference='https://llvm.org/docs/doxygen/AMDHSAKernelDescriptor_8h_source.html',descriptor_rsrc1_offset=48,F32_round_bit_offset=12,F32_denorm_bit_offset=16,flush_none=3,round_to_nearest_even=0)
