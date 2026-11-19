"""Fixed128B header and bounded independent zlib block contract. No payload at import."""
import hashlib,struct
from common import P,need
MAGIC=b'IVLZB01\n'
HEADER=struct.Struct('<8sIIQQII32s32s24s')

def raw_length(row,index):
    need(type(index)is int and 0<=index<row['blocks'],'BLOCK_INDEX_BOUND')
    return min(P['block_bytes'],row['bytes']-index*P['block_bytes'])

def pack(row,index,ncompressed,raw_digest,compressed_digest):
    nraw=raw_length(row,index);need(type(ncompressed)is int and 0<ncompressed<=nraw+65536,'COMPRESSED_BLOCK_CAP_NO_RAW_FALLBACK')
    need(len(raw_digest)==len(compressed_digest)==32,'BINARY_BLOCK_SHA_LENGTH')
    return HEADER.pack(MAGIC,1,row['file_id'],index,index*P['block_bytes'],nraw,ncompressed,raw_digest,compressed_digest,b'\0'*24)

def unpack(raw,row,index):
    need(len(raw)==HEADER.size==128,'EXACT_FIXED_HEADER');magic,version,file_id,i,offset,nraw,ncomp,hraw,hcomp,reserved=HEADER.unpack(raw)
    need(magic==MAGIC and version==1 and file_id==row['file_id'] and i==index and offset==index*P['block_bytes'] and reserved==b'\0'*24,'EXACT_VERSION_FILE_ORDER_OFFSET_RESERVED')
    need(nraw==raw_length(row,index) and 0<ncomp<=nraw+65536,'BOUNDED_DECLARED_LENGTH_BEFORE_BODY')
    return dict(raw_bytes=nraw,compressed_bytes=ncomp,raw_digest=hraw,compressed_digest=hcomp)

def decompress_bounded(comp,nraw):
    import zlib
    d=zlib.decompressobj(15);raw=d.decompress(comp,nraw+1)
    need(len(raw)==nraw and d.eof is True and d.unused_data==d.unconsumed_tail==b'','EXACT_BOUNDED_ZLIB_EOF_NO_TRAILING')
    return raw

def versions():
    import sys,zlib
    need(callable(zlib.compressobj) and callable(zlib.decompressobj) and zlib.DEFLATED==8,'STDLIB_ZLIB_CAPABILITY_METADATA')
    return dict(Python=sys.version,zlib_compile=zlib.ZLIB_VERSION,zlib_runtime=zlib.ZLIB_RUNTIME_VERSION,level=1,wbits=15,memLevel=8,strategy='Z_DEFAULT_STRATEGY',independent_blocks=True)
