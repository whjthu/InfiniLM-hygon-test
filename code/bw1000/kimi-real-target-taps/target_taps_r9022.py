"""Remote use only: retain raw feature/diagnostic buffers on project filesystem."""
import ctypes as C,json,pathlib
class Taps:
 def __init__(self,lib,context):
  self.lib,self.context=lib,context
  lib.modern_kimi_diag_capabilities_r9008.argtypes=[];lib.modern_kimi_diag_capabilities_r9008.restype=C.c_char_p
  lib.modern_kimi_diag_arm_r9008.argtypes=[C.c_void_p,C.c_char_p,C.c_int];lib.modern_kimi_diag_arm_r9008.restype=C.c_int
  lib.modern_kimi_diag_read_r9008.argtypes=[C.c_void_p];lib.modern_kimi_diag_read_r9008.restype=C.c_char_p
 def capabilities(self):return json.loads(self.lib.modern_kimi_diag_capabilities_r9008())
 def arm(self,label):
  if self.lib.modern_kimi_diag_arm_r9008(self.context,label.encode(),1):raise RuntimeError('tap arm failed')
 def read(self):return json.loads(self.lib.modern_kimi_diag_read_r9008(self.context))
 def close(self):
  if self.lib.modern_kimi_diag_arm_r9008(self.context,b'',0):raise RuntimeError('tap disable failed')
 @staticmethod
 def assemble(stage_records):
  import torch
  rows={r['layer']:r for stage in stage_records for r in stage['records'] if r['point']=='decoder'}
  layers=[19,37,66,78,90];selected=[rows[l] for l in layers]
  if any(r['positions']!=selected[0]['positions'] for r in selected):raise ValueError('target feature position alignment differs')
  tensors=[torch.frombuffer(bytearray(pathlib.Path(r['remote_owned_buffer']).read_bytes()),dtype=torch.bfloat16).reshape(r['shape']).clone() for r in selected]
  # This actual source assembly is not a claim of the checkpoint's training convention.
  return torch.cat(tensors,dim=-1),selected[0]['positions']
