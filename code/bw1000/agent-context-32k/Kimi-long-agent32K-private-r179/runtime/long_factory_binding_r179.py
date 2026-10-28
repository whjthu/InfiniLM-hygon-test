"""Declare the explicit new CAPI; no load/create/forward is executed here.
The future target loader must first pin the actual CPU build receipt, Native
SHA, old CAPI providers and dladdr for BOTH added functions to that same DSO.
"""
import ctypes as C
def declare_long_factory(lib):
 create=lib.modern_kimi_engine_create_long_agent_r179
 create.argtypes=[C.c_char_p,C.c_int,C.c_int,C.c_int,C.c_char_p,C.c_int];create.restype=C.c_void_p
 caps=lib.modern_kimi_long_agent_capabilities_r179;caps.argtypes=[C.c_void_p];caps.restype=C.c_char_p
 return create,caps
