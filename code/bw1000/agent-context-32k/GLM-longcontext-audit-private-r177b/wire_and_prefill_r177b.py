"""CPU-only planning contracts; not actual Native support or model correctness proof."""
MAGIC=743143
def admit(ids,output,context):
 if not ids or type(output)is not int or output<1 or len(ids)+output>context or any(type(x)is not int or not 0<=x<154880 for x in ids):raise ValueError('full official prompt plus output exceeds total context; no crop')
def encode(seq,ids,output,context):
 admit(ids,output,context);wire=[MAGIC,seq,1,len(ids),output]+list(ids);wire+=[0]*(context+8-len(wire));assert len(wire)==context+8 and all(type(x)is int and 0<=x<2**24 for x in wire);return wire
def decode(wire,seq,context):
 if len(wire)!=context+8 or any(int(x)!=x for x in wire):raise ValueError('exact FP32 integer transport')
 v=list(map(int,wire));magic,s,op,count,out=v[:5]
 if magic!=MAGIC or s!=seq or op!=1 or not 0<count<=context or any(v[5+count:]):raise ValueError('wire identity or zero transport tail')
 ids=v[5:5+count];admit(ids,out,context);return ids,out
def prefill_chunks(ids,chunk=512):
 if not ids or type(chunk)is not int or not 1<=chunk<=4096:raise ValueError('real Native forward tokens bound')
 result=[]
 for past in range(0,len(ids),chunk):result.append({'row':0,'past':past,'ids':list(ids[past:past+chunk]),'emit_first_generated_token':past+chunk>=len(ids)})
 assert [x for q in result for x in q['ids']]==ids and sum(q['emit_first_generated_token']for q in result)==1
 return result
