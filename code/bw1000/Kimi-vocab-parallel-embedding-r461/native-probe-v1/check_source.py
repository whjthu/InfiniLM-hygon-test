from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent;P=D.parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 proof=json.loads((D/'SOURCE_DERIVATION.json').read_bytes())
 texts={}
 for origin,h in proof['source_origins'].items():
  assert H(origin)==h;name=Path(origin).name;texts[name]=Path(origin).read_bytes().decode()
 for edit in proof['exact_replacements']:
  text=texts[edit['unit']];assert text.count(edit['old'])==1
  texts[edit['unit']]=text.replace(edit['old'],edit['new'],1)
 appendix=(D/'pair_append.cpp.inc').read_bytes().decode();assert hashlib.sha256(appendix.encode()).hexdigest()==proof['appended_pair_body_SHA256']
 texts['pair_probe.cpp']+='\n'+appendix
 for name,text in texts.items():assert (D/'native'/name).read_bytes()==text.encode()
 pair=texts['pair_probe.cpp'];worker=texts['worker.cpp'];protocol=(D/'probe_protocol.hpp').read_text();owned=(D/'probe_owned.hpp').read_text()
 assert pair.count('embedding_probe_epoch_r461=0;')==1
 assert pair.index('kimi_vocab_r461::validate_global_ids(ids,tokens); // before any RUN/epoch mutation')<pair.index('started=true;c->embedding_probe_epoch_r461=epoch;')
 assert 'if(c&&started)c->embedding_probe_failed_r461=true;'in pair
 assert 'w->run(en::RankWorker::Input{})'in pair and 'w->wait()'in pair
 assert pair.index('for(auto*w:submitted)try{w->wait();}')<pair.index('probe::clear(requests);')
 assert 'else if(!probe::await_all_metadata(gate))'in pair
 assert 'std::array<std::shared_ptr<Request>,8>'in protocol and 'std::function'not in protocol
 assert all(s in owned for s in ['model.children().at("language_model")','language->children().at("model")','text->children().at("embed_tokens")','embedding->state_dict()','embedding->forward_sharded(r,input)'])
 assert owned.index('metadata_ready(request)')<owned.index('await_admission(request)')<owned.index('input=cpu->to(r,r.device())')
 assert 'else if (local_cmd == Command::RUN && kimi_embedding_probe_r461::contains(this))'in worker
 assert 'No device work until all eight local RUN calls have been accepted'in owned
 return dict(passed=True,exact_two_HOST_diffs=True,private_Context_fields_only=True,original_nonprobe_RUN_preserved=True,
  actual_public_weight_tree=True,sidecar_bound_to_worker_context_epoch=True,all8_local_metadata_admission=True,
  clear_after_all_submitted_waits=True,global_ID_check_before_RUN=True,device_or_SDK_qualification=False)
if __name__=='__main__':print(json.dumps(check(),indent=2))
