def persist_memory_gate(label,packets,gate,report,save):
 values=[row[0]for p in packets for row in p.get('memory',[])];attempt={'label':label,'minimum_free_bytes':min(values)if values else None,'required_free_bytes':4*1024**3,'owners':packets};report['last_memory_guard_attempt']=attempt
 try:gate(packets)
 except BaseException:
  report['failed_memory_guard']=attempt
  try:save()
  except BaseException as error:report['failed_memory_guard_persist_error']=repr(error)
  raise
 report['last_memory_guard']=attempt;return attempt
