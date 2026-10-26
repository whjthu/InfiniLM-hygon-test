"""Failure-only evidence; retains the original memory-gate exception."""
import copy
GIB=1024**3
def failure_safe_save(report,save):
 try:return save()
 except BaseException as error:
  if 'failed_memory_guard'not in report:raise
  report.setdefault('memory_guard_evidence_save_errors',[]).append({'type':type(error).__name__,'error':repr(error)})
def gate_with_evidence(label,snapshot,packets,gate,report,save):
 try:return gate(packets) # unchanged one gate invocation on successful path
 except BaseException as original:
  evidence={'label':label,'required_free_bytes':4*GIB,'strict_greater_than_floor':True,'local_snapshot':copy.deepcopy(snapshot),'owners':copy.deepcopy(packets),'original_exception_type':type(original).__name__,'original_exception':repr(original)}
  try:evidence['minimum_free_bytes']=min(pair[0]for packet in packets for pair in packet['memory'])
  except BaseException as diagnostic:evidence['minimum_free_bytes']=None;evidence['diagnostic_error']=repr(diagnostic)
  report['failed_memory_guard']=evidence
  failure_safe_save(report,save)
  raise
