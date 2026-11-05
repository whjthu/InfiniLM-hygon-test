import ctypes as C,json
from types import SimpleNamespace
from binding import create,FACTORY,CAPS
CTX=0x1234567887654321;calls=[];errors=[];holders=[]
def factory(raw,node,address,port):
 try:
  q=json.loads(raw);assert q['infinilm_kimi_k192_v128_private']and node==3 and address==b'12.1.2.1'and port==28001;calls.append('actual4args');return CTX
 except BaseException as e:errors.append(repr(e));return 0
f=C.CFUNCTYPE(C.c_void_p,C.c_char_p,C.c_int,C.c_char_p,C.c_int)(factory);holders.append(f)
cap=C.CFUNCTYPE(C.c_void_p,C.c_void_p)(lambda ptr:0);holders.append(cap);lib=SimpleNamespace(**{FACTORY:f,CAPS:cap})
assert create(lib,{'infinilm_kimi_k192_v128_private':True},3,'12.1.2.1',28001)==CTX
try:create(lib,{},3,'12.1.2.1',28001);raise AssertionError('unselected accepted')
except ValueError:pass
assert not errors and calls==['actual4args']
print(json.dumps({'passed':True,'actual_binding4arg_high64':True,'Native_leaf':'recording','SDK_device_execution':False}))
