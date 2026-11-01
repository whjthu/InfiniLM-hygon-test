"""Explicit request-level chunk descriptor; math/cache/topology unchanged."""
ARMS=(1024,2048,1024,2048,2048,1024)
GPU_HARD_SECONDS=2700
WORK_SECONDS=2580
MEMORY_FLOOR_BYTES=16*1024**3
def descriptor_for_index(base,index):
 if type(index)is not int or not 0<=index<len(ARMS):raise ValueError('Fixed six Native requests exhausted')
 if base.get('prefill_chunks_supported')!=[1024,2048]or base.get('rows')!=1 or base.get('capacity')!=32768:raise ValueError('Worker/binding geometry differs')
 return dict(base,prefill_chunk=ARMS[index],arm_index=index,warmup=index<2,output_budget=128)
