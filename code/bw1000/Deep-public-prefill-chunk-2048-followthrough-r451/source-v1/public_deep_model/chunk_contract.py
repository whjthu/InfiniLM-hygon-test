"""Closed six-request chunk descriptor, no public-field policy selection."""
CHUNKS=(1024,2048,1024,2048,2048,1024)
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'

def descriptor_for_index(base,index):
    if type(index)is not int or not 0<=index<6:raise ValueError('Closed sixchunk plan index')
    if (base.get('library_SHA256'),base.get('rows'),base.get('capacity'),base.get('prefill_chunks_supported'))!=(NATIVE,1,32768,[1024,2048]):raise ValueError('Actual a314/one-row/32K binding differs')
    return dict(library_SHA256=NATIVE,rows=1,capacity=32768,prefill_chunks_supported=[1024,2048],prefill_chunk=CHUNKS[index],arm_index=index,warmup=index<2,output_budget=128,HC0=True,r366=0,fullpre=0,source_strategy='1257 alreadypositive/currentpublic followthrough only')
