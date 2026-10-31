"""Bounded CPU-side cleanup for the known owner session; no Native collectives."""
import os,signal,subprocess

def cleanup_owned(client,server,thread,engine,log,*,owned_pgid=None):
 q={'engine_stopped':False,'client_TERM_sent':False,'client_KILL_sent':False,'owned_group_TERM_sent':False,'owned_group_KILL_sent':False,'server_force_exit':False,'errors':[],'server_was_started':thread is not None and thread.ident is not None}
 try:
  if server is not None:server.should_exit=True
 except BaseException as e:q['errors'].append('server exit: '+repr(e))
 try:
  if engine is not None:engine.stop();q['engine_stopped']=True
 except BaseException as e:q['errors'].append('engine stop: '+repr(e))
 def group_signal(sig):
  if client is None or owned_pgid!=client.pid or type(owned_pgid)is not int or owned_pgid<=1:raise RuntimeError('Unproven owned process group')
  try:os.killpg(owned_pgid,sig)
  except ProcessLookupError:return False
  return True
 try:
  status=client.poll()if client is not None else 0
  if client is not None and(status is None or(owned_pgid is not None and status!=0)):
   if owned_pgid is None:client.terminate();q['client_TERM_sent']=True
   else:q['owned_group_TERM_sent']=group_signal(signal.SIGTERM)
   try:client.wait(timeout=30)
   except subprocess.TimeoutExpired:
    if owned_pgid is None:client.kill();q['client_KILL_sent']=True
   # A supervisor may exit on TERM while its phase child ignores TERM. Always
   # signal the same known group after TERM, even when the supervisor was reaped.
   if owned_pgid is not None:q['owned_group_KILL_sent']=group_signal(signal.SIGKILL)
   try:client.wait(timeout=10)
   except BaseException as e:q['errors'].append('client final wait: '+repr(e))
 except BaseException as e:q['errors'].append('client cleanup: '+repr(e))
 try:
  if thread is not None and thread.ident is not None:
   thread.join(timeout=30)
   if thread.is_alive():
    if server is not None:server.force_exit=True
    q['server_force_exit']=True;thread.join(timeout=10)
 except BaseException as e:q['errors'].append('HTTP thread cleanup: '+repr(e))
 try:
  if log is not None and not log.closed:log.close()
 except BaseException as e:q['errors'].append('client log close: '+repr(e))
 q['HTTP_thread_alive_after_cleanup']=thread is not None and thread.is_alive();q['forced_cleanup']=q['client_TERM_sent']or q['client_KILL_sent']or q['owned_group_TERM_sent']or q['owned_group_KILL_sent']or q['server_force_exit'];q['normal_server_shutdown']=q['server_was_started']and not q['HTTP_thread_alive_after_cleanup']and not q['forced_cleanup']and not q['errors'];return q
