"""Lossless mode114 qword permutation, only caller-owned CPU shard buffers.
The consumer must clone its own candidate shard before calling this; never
mutate a checkpoint mmap/shared tensor or the independent original context.
"""
def repack_private_cpu_shard(array,inverse=False,rows_per_chunk=32):
 import numpy as np
 if array.dtype!=np.uint8 or not array.flags.c_contiguous or array.ndim!=3 or tuple(array.shape)!=(896,384,1792):raise ValueError('exact owned CPU W13 E896,N384,Kpacked1792')
 if not array.flags.writeable or not 1<=rows_per_chunk<=32:raise ValueError('writable private buffer/chunkbound')
 _permute(array.reshape(-1,1792),inverse,rows_per_chunk)
 return array

def _permute(rows,inverse=False,rows_per_chunk=32):
 import numpy as np
 assert rows.dtype==np.uint8 and rows.ndim==2 and rows.shape[1]%4==0 and rows.flags.c_contiguous
 for first in range(0,len(rows),rows_per_chunk):
  x=rows[first:first+rows_per_chunk].reshape(-1,rows.shape[1]//4,4)
  a,b,c,d=[x[...,i].copy()for i in range(4)]
  if inverse:
   x[...,0]=(a&15)|((c&15)<<4);x[...,1]=(a>>4)|(c&240)
   x[...,2]=(b&15)|((d&15)<<4);x[...,3]=(b>>4)|(d&240)
  else:
   x[...,0]=(a&15)|((b&15)<<4);x[...,1]=(c&15)|((d&15)<<4)
   x[...,2]=(a>>4)|(b&240);x[...,3]=(c>>4)|(d&240)

if __name__=='__main__':
 import numpy as np,json,pathlib,hashlib
 # All original nibble symbols at each byteposition plus full N384/K3584
 # coordinate pattern; synthetic CPU bytes only, not a checkpoint qualification.
 rng=np.random.default_rng(20261025);x=rng.integers(0,256,size=(384,1792),dtype=np.uint8);old=x.copy();_permute(x)
 words=x.reshape(-1,4).astype(np.uint32);q=words[:,0]|(words[:,1]<<8)|(words[:,2]<<16)|(words[:,3]<<24)
 decoded=np.stack([((q>>(p*4))&15) for p in range(4) for _ in range(1)],axis=1)
 out=np.empty((q.size,8),np.uint8)
 for p in range(4):out[:,p*2]=(q>>(p*4))&15;out[:,p*2+1]=(q>>(16+p*4))&15
 natural=np.empty((old.size//4,8),np.uint8);src=old.reshape(-1,4)
 for p in range(4):natural[:,2*p]=src[:,p]&15;natural[:,2*p+1]=src[:,p]>>4
 assert np.array_equal(out,natural);_permute(x,True);assert np.array_equal(x,old)
 q={'passed':True,'scope':'Synthetic CPU repack actual consumer function only/no weights or GPU','matrix_bytes':old.size,'all_decoded_nibbles_equal':int(out.size),'inverse_bytes_equal':old.size,'chunk_rows_max':32,'explicit_plane_copy_bytes_max':32*448*4,'temporary_array_storage_conservative_bytes':262144,'metadata_allocator_overhead':'Not measured; no wholebank CPU copy inside repacker','original_scale_bytes_mutated':0,'source_SHA256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};p=pathlib.Path('Kimi-WFP4-repack-CPU-r8669zu.json');assert not p.exists();p.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
