"""Read original local process creation identities; never signal or restart them."""
from pathlib import Path
import ctypes as C,datetime,hashlib,json
from ctypes import wintypes as T
D=Path(__file__).resolve().parent;old=D.parent/'training-vggt-original-public-model-v1'
out=json.loads((old/'transfer-outcome.json').read_bytes());owned=json.loads((old/'owned-local-transports.json').read_bytes())
assert out['passed'] and out['source_rc']==out['destination_rc']==0 and out['error_type'] is None and out['request_id']==owned['request_id']
k=C.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[T.DWORD,T.BOOL,T.DWORD];k.OpenProcess.restype=T.HANDLE
k.GetProcessTimes.argtypes=[T.HANDLE,C.POINTER(T.FILETIME),C.POINTER(T.FILETIME),C.POINTER(T.FILETIME),C.POINTER(T.FILETIME)];k.GetProcessTimes.restype=T.BOOL;k.CloseHandle.argtypes=[T.HANDLE]
rows={}
for name in ('driver','sender','receiver'):
 pin=owned[name];handle=k.OpenProcess(0x1000,False,pin['PID'])
 if not handle:
  error=C.get_last_error();assert error==87
  row=dict(PID=pin['PID'],expected_creation_FileTimeUTC=pin['startcreationticks'],original_identity_present=False,state='PID_MISSING')
 else:
  values=[T.FILETIME() for _ in range(4)]
  try:assert k.GetProcessTimes(handle,*[C.byref(v) for v in values])
  finally:k.CloseHandle(handle)
  current=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime
  row=dict(PID=pin['PID'],expected_creation_FileTimeUTC=pin['startcreationticks'],current_creation_FileTimeUTC=current,original_identity_present=current==pin['startcreationticks'],state='LIVE_IDENTITY_MATCH' if current==pin['startcreationticks'] else 'PID_REUSED_DIFFERENT_IDENTITY')
 assert not row['original_identity_present'];rows[name]=row
result=dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_request_id=out['request_id'],processes=rows,all_original_local_processes_absent=True,source_rc=out['source_rc'],destination_rc=out['destination_rc'],original_transfer_outcome_SHA256=hashlib.sha256((old/'transfer-outcome.json').read_bytes()).hexdigest(),no_signal_or_restart=True)
with (D/'original-local-process-terminal.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(original_local_processes_absent=True,source_rc=0,destination_rc=0)))
