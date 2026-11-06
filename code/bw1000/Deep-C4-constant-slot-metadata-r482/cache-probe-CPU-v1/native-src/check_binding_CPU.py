"""Actual ctypes high64 recording only: no Native library or GPU is loaded."""
import ctypes as C
import numpy as np
from binding import API
class Library:pass
lib=Library();handle=(1<<60)+0x12345;seen=[];callbacks=[]
def add(name,args,ret,func):
 cb=C.CFUNCTYPE(ret,*args)(func);callbacks.append(cb);setattr(lib,'deep_slot_probe_'+name+'_r482',cb)
def check(h):assert h==handle;seen.append(h)
add('error',[],C.c_char_p,lambda:b'recording')
add('create',[C.c_int],C.c_void_p,lambda mode:handle)
def seed(h,layer,begin,n,v,s):check(h);assert(layer,begin,n)==(20,0,3);return 0
add('seed',[C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t,C.c_void_p,C.c_void_p],C.c_int,seed)
def gather(h,layer,ids,n,v,nb,s,ns,clock,nc):
 check(h);assert layer==20 and n==3 and nb==384 and ns==3 and nc==3;assert list((C.c_int64*n).from_address(ids))==[0,63,8191];C.memset(v,0,nb);C.memset(s,0,ns*4);return 0
add('gather',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_int,gather)
add('guard',[C.c_void_p,C.c_size_t,C.c_int],C.c_int,lambda h,l,k:(check(h),0)[1])
add('reset',[C.c_void_p,C.c_size_t],C.c_int,lambda h,l:(check(h),0)[1])
def raw(h,layer,p,n):check(h);assert n==128*8448;return 0
add('raw',[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_int,raw)
add('observe',[C.c_void_p,C.c_void_p,C.c_size_t],C.c_int,lambda h,p,n:(check(h),0)[1])
def destroy(h,p,n):check(h);assert n==4;(C.c_uint64*4).from_address(p)[2]=1;return 0
add('destroy',[C.c_void_p,C.c_void_p,C.c_size_t],C.c_int,destroy)
a=API(lib,0);a.seed(20,0,np.zeros((3,128),np.uint8),np.ones((3,1),np.float32));v,s,_=a.gather(20,np.array([0,63,8191],np.int64));assert v.shape==(3,128)and s.shape==(3,1);a.guard(20,0);a.reset(20);a.raw(20);a.observe();a.close();assert a.closed['values'][2]==1 and len(seen)==7;print('PASS high64 ctypes seven callback paths/outputcanary bounds; no Native/SDK/GPU')
