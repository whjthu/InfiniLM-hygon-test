"""Offline stdlib audit of original920 receipt/object/ELF/ISA/metadata bytes.

No Runtime, Native, Slurm, compiler, device or third-party module is executed.
This file is a later read-only consumer, outside the immutable CPU99f0 source MF.
"""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,re,struct
def audit(path):
 seal=json.loads(Path(path).read_bytes());assert seal['actual_CPU']=='910407.920'and seal['GPU']==0
 files={};hashes={}
 for row in seal['files']:
  name=Path(row['path']).name;assert name not in files
  raw=gzip.decompress(base64.b64decode(row['gzip_base64']));assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['SHA256'];files[name]=raw;hashes[name]=row['SHA256']
 assert hashes['CPU-build.json']=='91b4eb272112ca967c721467dd9205f2a34836adfa7c8a4d77d5165c602c60ba'
 q=json.loads(files['CPU-build.json']);assert q['completed']and q['passed']and q['producer']=='910407.920'
 assert q['GPU']==0 and not q['Native_dlopen']and not q['Runtime_executed']and not q['weights_opened']
 assert q['source_manifest_SHA256']=='99f0d05bdb75ad5c9e7fbfb3307290296505c4646eb3cd8d6ba688c3c31724fa'
 assert q['HOST_TUs_compiled']==q['GPU_TUs_compiled']==1 and len(q['reused_objects_SHA256'])==29 and len(q['final_objects_SHA256'])==31
 expected=dict(q['reused_objects_SHA256'])
 for name in ['compiled_HOST','compiled_GPU']:
  row=q[name];expected[row['object']]=row['object_SHA256'];assert hashes[Path(row['MD']).name]==row['MD_SHA256']
 assert expected==q['final_objects_SHA256']and hashes['fp32_leaves_r446.o']==q['compiled_GPU']['object_SHA256']
 elf=files['GPU-image-0.elf'];obj=files['fp32_leaves_r446.o'];assert obj.count(elf)==1 and elf[:6]==b'\x7fELF\x02\x01'and struct.unpack_from('<H',elf,18)[0]==224
 base=struct.unpack_from('<Q',elf,40)[0];width,count=struct.unpack_from('<HH',elf,58);assert width==64 and base+count*64<=len(elf)
 sections=[struct.unpack_from('<IIQQQQIIQQ',elf,base+i*64)for i in range(count)];descs={}
 for section in sections:
  if section[1]not in [2,11]:continue
  assert section[9]==24;strings=sections[section[6]];names=elf[strings[4]:strings[4]+strings[5]]
  for off in range(section[4],section[4]+section[5],24):
   name,info,other,index,value,size=struct.unpack_from('<IBBHQQ',elf,off);end=names.find(b'\0',name);assert end>=0;symbol=names[name:end].decode()
   if not symbol.endswith('.kd')or not any(x in symbol for x in ['cast_kernel','gemv_kernel']):continue
   target=sections[index];at=target[4]+value-target[3];raw=elf[at:at+64];assert len(raw)==64
   bits=int.from_bytes(raw[48:52],'little');assert(bits>>12)&3==0 and(bits>>16)&3==3;descs[symbol]=dict(offset=at,round32=0,denorm32=3,raw_hex=raw.hex())
 assert len(descs)==3 and sum('gemv_kernel'in x for x in descs)==2
 isa=files['GPU0.ISA.log'].decode();notes=files['GPU0.notes.log'].decode();assert "amdgcn-amd-amdhsa--gfx936"in notes
 assert not re.search(r'\bv_(?:mfma|mmac)[_a-z0-9]*',isa,re.I)
 fma=re.findall(r'\bv_fma(?:c)?_f32(?:_e(?:32|64))?\b',isa);assert fma
 c=q['actual_new_GPU_object_inspection'];assert c['passed']and len(c['images'])==1
 image=c['images'][0];assert image['SHA256']==hashes['GPU-image-0.elf']and image['ISA_log_SHA256']==hashes['GPU0.ISA.log']and image['notes_log_SHA256']==hashes['GPU0.notes.log']
 for d in image['descriptors']:assert descs[d['symbol']]['raw_hex']==d['descriptor_hex']
 pure=json.loads(files['bits.run.log']);assert pure['passed']and pure['finite_BF16_encodings']==65280 and not pure['GPU']and not pure['Runtime']
 parse=lambda raw:{line.split()[-1]:line.split()[-2]for line in raw.decode().splitlines()if len(line.split())>=3}
 old,new=parse(files['parent.nm.log']),parse(files['candidate.nm.log']);assert all(new.get(n)==t for n,t in old.items()if t not in ['W','V','u','w','v']or not n.startswith('_Z'))
 for n in ['glm_value_cast_r446','glm_value_gemv_r446','glm_attention_value_fullfp32_r446']:assert new[n]=='T'
 assert b'not found'not in files['candidate.ldd.log']and b'undefined symbol:'not in files['candidate.ldd.log']
 return dict(passed=True,actual_CPU='910407.920',files=len(files),file_SHA256=hashes,new_GPU_object_embeds_exact_ELF=True,scalar_F32_FMA_opcode_count=len(fma),kernel_descriptors=descs,actual_target='gfx936:sramecc+',finite_BF16_CPU_cases=65280,source_manifest_SHA256=q['source_manifest_SHA256'],library_SHA256=q['library_SHA256'],GPU=False,Runtime=False,device_numeric=False,Model_READY=False,scope='Original artifact/ISA/descriptor bytes independently checked; no actual device mathematical or complete attention qualification')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('seal');a.add_argument('--output');args=a.parse_args();q=audit(args.seal);raw=json.dumps(q,indent=2)+'\n'
 if args.output:Path(args.output).write_text(raw,encoding='utf8')
 print(raw)
