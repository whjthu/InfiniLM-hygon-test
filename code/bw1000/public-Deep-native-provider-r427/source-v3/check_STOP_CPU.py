"""Only changed nonblocking active STOP bridge/Run/watcher; Native leaf records drain."""
import json,tempfile,threading,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from public_deep_native.actors import Actors
from public_deep_native.protocol import Run
from public_deep_native.cli import watch_public_shutdown
from public_deep_native.prepared_queue import prepare_engine
import enum
class Reason(str,enum.Enum):
 CANCELED='canceled';ERROR='error';TIMEOUT='timeout';EOS_TOKEN='eos_token';LENGTH='length'
class ProcessorLeaf:
 def get_tokenizer(self):return self
 def apply_chat_template(self,messages,add_generation_prompt=False,**kw):return json.dumps(messages)+('<assistant>'if add_generation_prompt else'')
 def __call__(self,prompt,**kw):return {'input_ids':[2,3]}
 def decode(self,ids,**kw):return 'recorded'
def run():
 with tempfile.TemporaryDirectory()as tmp:
  root=Path(tmp);(root/'generation_config.json').write_text('{"eos_token_id":1}')
  config=dict(runtime_directory=tmp,run='actual-active-stop',actual_step='CPU.recording',source_manifest_SHA256='CPU',provider_config_SHA256='CPU-config',gpu_start_epoch=time.time(),Native_seconds=2580,service_seconds=2400)
  prepared=prepare_engine('Deep',str(root),processor=ProcessorLeaf(),finish_reasons=Reason);engine=prepared['engine'];engine.start()
  req=engine.add_chat_request(messages=[dict(role='user',content='full source STOP test')],sampling_params=SimpleNamespace(temperature=0.,max_tokens=4096,stop=None),request_id='active',request_data={'max_tokens':4096,'_agent_schemas':{},'_agent_choice':'none'})
  entered,release,drained=threading.Event(),threading.Event(),threading.Event()
  def Native_owner_leaf(config,prepared,stop_intent):
   actual=engine.take(timeout=2);assert actual is req;entered.set();assert release.wait(3)
   assert req.cancel_event.is_set()and not req.is_finished()and stop_intent.is_set()
   drained.set();req.finish(Reason.CANCELED)
   Run(config).write('closed-engine.json',dict(normal=True,engine_destroy_returned_zero=True))
  with patch('public_deep_native.actors.owner_main',Native_owner_leaf):
   group=Actors(config,prepared);group.start();assert entered.wait(2);group.run.stop();http=SimpleNamespace(should_exit=False)
   watch_public_shutdown(SimpleNamespace(group=group),http,threading.Event(),time.time()+100)
   assert http.should_exit and group.stop_intent.is_set()and req.cancel_event.is_set()
   assert group.owner_thread.is_alive()and not req.is_finished()and not drained.is_set()
   assert not engine.is_healthy()and not group.failed.is_set()
   release.set();group.stop();assert drained.is_set()and req.finish_reason==Reason.CANCELED and not group.owner_thread.is_alive()
 return dict(passed=True,case='actual_Run_STOP_to_watch_request_stop_bridge_cancel_then_Native_drain_and_bounded_join',Native_leaf='Only Native owner call/drain/destroy is recording; actual Actors/Run/prepared admission/NativeRequest/watcher executed',active_not_finished_before_drain=True,watcher_did_not_join=True,GPU=False,Native_dlopen=False,jobs=0,old_three_cases_repeated=False)
if __name__=='__main__':
 q=run();(Path(__file__).parent/'stop-CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
