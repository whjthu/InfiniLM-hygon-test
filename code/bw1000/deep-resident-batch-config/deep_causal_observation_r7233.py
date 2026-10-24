"""Observation-only watcher helpers; unknown is never terminal."""
import subprocess
TERMINAL=frozenset(['COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED','SPECIAL_EXIT'])
def terminal_state(value):
 token=value.strip().split()[0].rstrip('+')if value.strip()else ''
 return token if token in TERMINAL else None

def observed_run(args,timeout,runner=None):
 if runner is None:runner=subprocess.run
 try:
  p=runner(args,capture_output=True,text=True,timeout=timeout)
  return {'observed':True,'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-4000:]}
 except subprocess.TimeoutExpired as e:
  return {'observed':False,'status':'UNKNOWN','kind':'TimeoutExpired','timeout_seconds':timeout,'error':str(e),'no_resubmit':True}
 except (OSError,subprocess.SubprocessError)as e:
  return {'observed':False,'status':'UNKNOWN','kind':type(e).__name__,'error':str(e),'no_resubmit':True}
