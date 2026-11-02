"""Fixed1024 chunks, independent default-OFF HC T1 mode schedule."""
ARMS=(1024,1024,1024,1024,1024,1024)
HC_MODES=(0,1,0,1,1,0)
GPU_HARD_SECONDS=2700
WORK_SECONDS=2580
MEMORY_FLOOR_BYTES=16*1024**3
def descriptor_for_index(base,index):
 if type(index)is not int or not 0<=index<6:raise ValueError('Fixed six Native requests exhausted')
 if base.get('prefill_chunks_supported')!=[1024,2048]or base.get('rows')!=1 or base.get('capacity')!=32768 or base.get('hc_epilogue_modes_supported')!=[0,1]:raise ValueError('Worker/binding HC geometry differs')
 return dict(base,prefill_chunk=1024,hc_epilogue_mode=HC_MODES[index],arm_index=index,warmup=index<2,output_budget=128)
