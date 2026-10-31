"""Bounded cleanup that preserves an active owner failure and never skips the HTTP/queue close."""
import subprocess

def cleanup_owned(client,server,thread,engine,log):
 q={'engine_stopped':False,'client_TERM_sent':False,'client_KILL_sent':False,'server_force_exit':False,'errors':[],'server_was_started':thread.ident is not None}
 try:server.should_exit=True
 except BaseException as e:q['errors'].append('server exit: '+repr(e))
 try:engine.stop();q['engine_stopped']=True
 except BaseException as e:q['errors'].append('engine stop: '+repr(e))
 try:
  if client is not None and client.poll()is None:
   client.terminate();q['client_TERM_sent']=True
   try:client.wait(timeout=30)
   except subprocess.TimeoutExpired:
    client.kill();q['client_KILL_sent']=True
    try:client.wait(timeout=10)
    except BaseException as e:q['errors'].append('client KILL wait: '+repr(e))
 except BaseException as e:q['errors'].append('client cleanup: '+repr(e))
 try:
  if thread.ident is not None:
   thread.join(timeout=30)
   if thread.is_alive():server.force_exit=True;q['server_force_exit']=True;thread.join(timeout=10)
 except BaseException as e:q['errors'].append('HTTP thread cleanup: '+repr(e))
 try:
  if log is not None and not log.closed:log.close()
 except BaseException as e:q['errors'].append('client log close: '+repr(e))
 q['HTTP_thread_alive_after_cleanup']=thread.is_alive();q['forced_cleanup']=q['client_KILL_sent']or q['server_force_exit'];q['normal_server_shutdown']=q['server_was_started']and not q['HTTP_thread_alive_after_cleanup']and not q['forced_cleanup']and not q['errors'];return q
