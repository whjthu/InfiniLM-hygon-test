"""Same r9004 fixture/matrix, comparing chunk Math0 with KDA-only T1 calls."""
import argparse,ctypes,hashlib,json,pathlib,time,traceback
import lifecycle_base_r9020 as base

p=argparse.ArgumentParser();p.add_argument('--gpu-lib',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--root-go',action='store_true');a=p.parse_args()
assert a.root_go and not a.out.exists()
q={'status':'FAILED','Model_final_IDs_EOS':'NOT_TESTED','Model675_cause':'NOT_ISOLATED','selectors':{},'no_GPU_F64':True}
failed=None
try:
 for selector in ['kda_conv_run','kda_conv_run_serial_kda']:
  native=None;error=None
  d={'closed_normally':False}
  try:
   native=base.Native(a.gpu_lib,0)
   fn=getattr(native.lib,selector);fn.argtypes=native.lib.kda_conv_run.argtypes
   native.entry=fn
   # Explicit selector binding is verified against the actually loaded candidate.
   class Info(ctypes.Structure):
    _fields_=[('file',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
   dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)]
   info=Info();assert dl.dladdr(ctypes.cast(fn,ctypes.c_void_p),ctypes.byref(info))
   actual=pathlib.Path(info.file.decode());assert actual.resolve()==a.gpu_lib.resolve()
   d['selector_binding']={'symbol':selector,'file':str(actual),'SHA256':hashlib.sha256(actual.read_bytes()).hexdigest()}
   start=time.perf_counter();d['exercise']=base.exercise(native);d['wholechain_wall_s']=time.perf_counter()-start
   d['operation_counts_conv_silu_kda_tokens']=native.counts()
   assert d['operation_counts_conv_silu_kda_tokens']==([1143,1143,1143,1297]if selector=='kda_conv_run'else[1143,1143,1297,1297])
   assert len(d['exercise']['cases'])==80 and d['exercise']['missing_replay_negative_controls_detected']==35
   d['diagnostic_maxima']={key:max(x[key]['max_abs']for x in d['exercise']['cases']if x['continuation'])for key in ['KDA_state_diagnostic','continuation_output_diagnostic','continuation_state_diagnostic','verify_allposition_diagnostic']}
   d['original_symbol_binding']=native.binding
  except BaseException as e:
   error=e;d['exception']=repr(e);d['traceback']=traceback.format_exc()
   if native:
    try:d['operation_counts_conv_silu_kda_tokens']=native.counts()
    except BaseException as ce:d['count_error']=repr(ce)
  finally:
   if native:
    try:native.close();d['closed_normally']=True
    except BaseException as ce:
     d['close_error']=repr(ce)
     if error is None:error=ce
   q['selectors'][selector]=d
   if error is not None:raise error
 maps=pathlib.Path('/proc/self/maps').read_text()
 libs=sorted({line.split()[-1]for line in maps.splitlines()if any(n in line for n in ['libinfiniops.so','libinfinirt.so','libinfinilm_runtime.so'])})
 q['actual_runtime_dependencies_SHA256']={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()for p in libs}
 q['status']='COMPONENT_DIAGNOSTIC_EXECUTED_NOT_MODEL_QUALIFIED'
except BaseException as e:failed=e;q['exception']=repr(e);q['traceback']=traceback.format_exc()
finally:
 q['source_SHA256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();a.out.write_text(json.dumps(q,indent=2))
 print(json.dumps({k:v for k,v in q.items()if k!='selectors'}))
if failed is not None:raise failed
