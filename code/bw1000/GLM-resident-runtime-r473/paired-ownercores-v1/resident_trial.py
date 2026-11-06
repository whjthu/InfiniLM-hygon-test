"""Frozen one-load runtime arm order and passive existing six-scalar profile.
No Native/SDK/Runtime/Torch is imported or loaded by this module.
"""
import ctypes as C,math
PROFILE_NAMES=('incoming_host_s','layers_host_s','outgoing_host_s','incoming_stream_s','layers_stream_s','outgoing_stream_s')
def schedule():
 # Warm both complete arms, then3 paired units AB/BA/AB. Pair1 uses full MMLU,
 # the others full GSM. Repeated two questions are timing probes, not widequality.
 order=[('A',0,True,-1),('B',0,True,-1),('A',1,True,-1),('B',1,True,-1),('A',0,False,0),('B',0,False,0),('B',1,False,1),('A',1,False,1),('A',0,False,2),('B',0,False,2)]
 return [dict(index=i,arm=a,case_index=c,warmup=w,pair=p,owner_cores=4 if a=='A'else 2,prefill_chunk=512,phase='case'+str(i))for i,(a,c,w,p)in enumerate(order)]
def phases():return ['created','loaded']+[x['phase']for x in schedule()]+['quiescent','closed']
def owner_mask(common,device,cores):
 if type(device)is not int or not 0<=device<8 or cores not in(2,4)or len(common)<64 or len(set(common))!=len(common):raise ValueError('original64CPU block/owner2or4')
 mask=list(common[device*8:device*8+cores])
 if len(mask)!=cores:raise ValueError('incomplete owner CPUmask')
 return mask

def attach_profile(library):
 # This getter returns a thread_local six-double array owned by the sameNative.
 # Copy immediately on the existing ownerthread, before any next Nativeforward.
 f=library.modern_glm_pipeline_last_profile;f.argtypes=[];f.restype=C.POINTER(C.c_double)
 return f

def read_profile(f):
 p=f()
 if not p:raise RuntimeError('null profile pointer')
 v=[float(p[i])for i in range(6)]
 if any(not math.isfinite(x)or x<0 for x in v):raise RuntimeError('nonfinite/negative existing profile')
 return v

def add_profile(bucket,values,tokens,past):
 if len(values)!=6 or any(not math.isfinite(x)or x<0 for x in values):raise ValueError('six finite scalars')
 if not bucket:bucket.update(count=0,sum_s=[0.]*6,min_s=list(values),max_s=list(values),selected=[])
 bucket['count']+=1
 for i,x in enumerate(values):bucket['sum_s'][i]+=x;bucket['min_s'][i]=min(bucket['min_s'][i],x);bucket['max_s'][i]=max(bucket['max_s'][i],x)
 if tokens>1 or bucket['count']in(1,2,3,4,8,16,32,64,128,256,512,1024,2048,4096):
  if len(bucket['selected'])<24:bucket['selected'].append(dict(index=bucket['count'],tokens=tokens,past=past,seconds=values))
