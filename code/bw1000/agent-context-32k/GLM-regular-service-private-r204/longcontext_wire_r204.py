CONTEXT=32768
OUTPUT_MAX=8192
WIRE=CONTEXT+8
MAGIC=743143
WAIT,BEGIN,STOP=0,1,2
def prefill_chunks(ids,chunk=512):
 if not ids or chunk!=512:raise ValueError('Complete nonempty prompt and512 chunk required')
 return [(i,list(ids[i:i+chunk]))for i in range(0,len(ids),chunk)]
def encode_wire(seq,op,ids=(),limit=0):
 if op not in[WAIT,BEGIN,STOP]or len(ids)>CONTEXT:raise ValueError('Native wire context')
 if op==BEGIN and(not ids or type(limit)is not int or not 1<=limit<=OUTPUT_MAX or len(ids)+limit>CONTEXT):raise ValueError('Complete context/output budget; no truncation')
 if op!=BEGIN and(ids or limit):raise ValueError('Control payload')
 values=[MAGIC,seq,op,len(ids),limit]+list(ids);values+=[0]*(WIRE-len(values))
 if any(type(x)is not int or not 0<=x<2**24 for x in values):raise ValueError('Exact FP32 integer wire')
 return values
def decode_wire(values,seq,vocab):
 if len(values)!=WIRE or any(int(x)!=x for x in values):raise ValueError('Wire shape/type')
 values=list(map(int,values));magic,s,op,count,limit=values[:5]
 if magic!=MAGIC or s!=seq or op not in[WAIT,BEGIN,STOP]or not 0<=count<=CONTEXT:raise ValueError('Wire identity')
 if any(values[5+count:])or any(not 0<=x<vocab for x in values[5:5+count]):raise ValueError('Wire tokens/tail')
 prompt=values[5:5+count];encode_wire(seq,op,prompt,limit)
 return op,prompt,limit
