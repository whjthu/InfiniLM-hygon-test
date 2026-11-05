"""Exact new opt-in CABI only; old create ABI is inherited byte-exact."""
import ctypes

def declare_trace(lib):
 f=lib.glm_global_tp16_trace_path_r470;f.argtypes=[ctypes.c_void_p,ctypes.c_char_p];f.restype=ctypes.c_int

def check_new_ffi():
 high=0x1234567800000042;seen=[];callback=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_char_p)(lambda ctx,path:seen.append((ctx,path.decode()))or 0)
 class Leaf:pass
 leaf=Leaf();leaf.glm_global_tp16_trace_path_r470=callback;declare_trace(leaf);assert leaf.glm_global_tp16_trace_path_r470(high,b'/local/r470-passive-recording.jsonl')==0 and seen==[(high,'/local/r470-passive-recording.jsonl')];return dict(passed=True,high_ctx=high,new_CABI=1,recording_leaf=True,Native=False,Runtime=False,GPU=False)
