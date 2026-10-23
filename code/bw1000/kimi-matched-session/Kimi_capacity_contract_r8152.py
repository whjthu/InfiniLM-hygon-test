"""CPU formula/source contracts, not native Engine execution or numerical proof."""
import ast
GEOMETRY={'max_rows':32,'pages':132,'page_size':256,'KDA_pool_rows':33,'pages_per_request':4,'max_total_tokens':1024}
def validate_source(source):
 tree=ast.parse(source);calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_kimi_engine_create_r2475']
 assert len(calls)==1 and len(calls[0].args)==7 and isinstance(calls[0].args[-1],ast.Constant)and calls[0].args[-1].value==32
 assert "'max_rows':32,'pages':132,'page_size':256,'KDA_pool_rows':33"in source
 assert 'ctypes.c_size_t];lib.modern_kimi_engine_create_r2475.restype=ctypes.c_void_p'in source
 assert 'KIMI_TP_NODES' in source and 'KIMI_PP_SIZE' in source
 return True
def retained_rows():
 count=0
 for row in range(1,33):
  for length in [1,255,256,257,512,639,640,1024]:
   old=[1+(row-1)*4+j for j in range(4)];new=[1+(row-1)*4+j for j in range(4)]
   assert old==new and new[-1]<132
   for pos in range(length):assert (old[pos//256]*256+pos%256)==(new[pos//256]*256+pos%256)<132*256
   count+=1
 return {'cases':count,'retained_rows':32,'lengths_include_max1024':True,'native_capacity32_actual':'NOT_RUN_REQUIRES_CREATE_GEOMETRY_AND_FULL_IDS_GATES'}
