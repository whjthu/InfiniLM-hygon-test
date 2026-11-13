"""Frozen one-load runtime arm order and passive existing six-scalar profile.
No Native/SDK/Runtime/Torch is imported or loaded by this module.
"""
import ctypes as C,math
PROFILE_NAMES=('incoming_host_s','layers_host_s','outgoing_host_s','incoming_stream_s','layers_stream_s','outgoing_stream_s')
def schedule():
 from observer_trial import schedule as screen_schedule
 return[dict(a,arm='diagnostic'if a['diagnostic']else'screen',pair=-1,warmup=False)for a in screen_schedule()]
def phases():
 from observer_trial import phases as screen_phases
 return screen_phases()

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
