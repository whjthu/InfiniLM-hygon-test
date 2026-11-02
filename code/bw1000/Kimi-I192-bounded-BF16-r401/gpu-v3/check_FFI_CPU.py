"""Actual payload builder/argtypes/ctypes conversion with real CPU buffer addresses.

The cdecl callback records ABI only; it is not the private Native or a GPU Runtime.
"""
from pathlib import Path
import ctypes
import json
import tempfile
import time
import types
from plan import records, descriptor, pointer_payload
from GPU_run import CAPI_ARGTYPES, bind_complete_capi, execute_calls


def check():
    holders = [ctypes.create_string_buffer(16) for _ in range(16)]
    class View:
        def __init__(self, buffer): self.buffer = buffer
        def data_ptr(self): return ctypes.addressof(self.buffer)
    matrices = []
    for fixture in range(2):
        buffers = [View(b) for b in holders[8 * fixture:8 * fixture + 8]]
        matrices.append(pointer_payload(*buffers[:4], buffers[4:], types.SimpleNamespace(cuda_stream=0)))
    assert all(v[-1] == 0 and all(p > 0 for p in v[:8]) for v in matrices)
    calls = []; mode = [0]; counters = dict(metadata=[0]*7, decoder=[0]*4)
    Callback = ctypes.CFUNCTYPE(ctypes.c_int, *CAPI_ARGTYPES)
    def leaf(*args):
        assert list(args[:7]) == [2,0,512,896,3584,192,16]
        assert args[-1] is None  # True ctypes void* conversion of default stream0.
        expected = matrices[len(calls) // 34]
        assert list(args[7:15]) == expected[:8]
        calls.append(args)
        m = counters['metadata']; m[0] += 1; m[1] += 512; m[2] += 3; m[3] += 9
        if mode[0]:
            v = counters['decoder']; v[0] += 3; v[1] += 33; v[2] += 3; v[3] += 33
        return 0
    fn = bind_complete_capi(Callback(leaf))
    def counter(): return {k:list(v) for k,v in counters.items()}
    def set_mode(a): mode[0]=a; return 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'actual-FFI-result.json'
        def observe(rows):
            path.write_text(json.dumps(dict(rows=rows, completed_CAPI=len(calls)), allow_nan=False))
        rows = execute_calls(matrices,time.time()+30,set_mode,lambda args:fn(*args),counter,
                             lambda r:dict(finite=True,numeric_pass=True,relative_RMS=0,max_abs=0,
                                           maxabs_over_reference_max=0,readonly=True,output_tail_canary=True),observe)
        saved = json.loads(path.read_bytes())
        assert saved['completed_CAPI']==len(saved['rows'])==len(rows)==68 and all(v['passed']for v in rows)
    rejected = []
    for label, value in [('data0',[0]+matrices[0][1:]),('negative_stream',matrices[0][:8]+[-1]),
                         ('bad_count',matrices[0][:8])]:
        try: descriptor(list(records())[0],value,time.time()+30)
        except ValueError: rejected.append(label)
        else: raise AssertionError('Invalid ABI data accepted')
    assert len(calls)==68 and len(rejected)==3
    return dict(completed=True,passed=True,actual_main_pointer_builder=True,actual_production_argtypes=True,
                ctypes_CFUNCTYPE_cdecl_recording=True,real_CPU_buffers=16,stream0_converts_to_NULL=True,
                actual_execute_calls=68,real_file_saved=True,invalid_before_CAPI=rejected,
                GPU=False,Runtime=False,Slurm=False,private_DSO_dlopen=False,
                scope='Actual8data+streambuilder/ctypes ABI/consumer recording only. No numerical or GPU proof.')


if __name__=='__main__': print(json.dumps(check()))
