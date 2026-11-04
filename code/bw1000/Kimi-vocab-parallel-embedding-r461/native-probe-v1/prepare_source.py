"""Assemble two isolated HOST TUs from CPU981/966 sealed source bytes.

Local use takes explicit archive paths. Does not execute a candidate, compile,
load an ELF, read checkpoint payloads or contact the cluster.
"""
from pathlib import Path
import base64,gzip,hashlib,json,sys
D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def archive(path,expected):
 t=json.loads(Path(path).read_bytes());packed=base64.b64decode(t['gzip_base64']);assert H(packed)==expected==t['archive_SHA256']
 raw=gzip.decompress(packed);assert H(raw)==t['uncompressed_SHA256'];a=json.loads(raw)
 for p,v in a['files'].items():assert H(v['rawtext'].encode())==v['SHA256'],p
 return a['files']
def assemble(new,old):
 pair_path='/refactor/incoming/Kimi-vocab-parallel-embedding-r461/cpu-repair-v2/native/pair_probe.cpp'
 worker_path='/refactor/incoming/Kimi-AR-persistent-storage-optin-r456/native/worker.cpp'
 pair=new[pair_path]['rawtext'];worker=old[worker_path]['rawtext'];changes=[]
 def replace(source,needle,value,unit):
  assert source.count(needle)==1,(unit,needle)
  changes.append(dict(unit=unit,old=needle,new=value));return source.replace(needle,value,1)
 pair=replace(pair,'#include "embedding_index_r461.hpp"','#include "probe_protocol.hpp"\n#include "embedding_index_r461.hpp"','pair_probe.cpp')
 pair=replace(pair,'struct ProbeEngine:en::InferEngine{','struct ProbeEngine:en::InferEngine{void embedding_probe_r461(const void*,std::uint64_t,const std::int64_t*,std::size_t,std::uint16_t*);','pair_probe.cpp')
 pair=replace(pair,'struct Context{','struct Context{std::uint64_t embedding_probe_epoch_r461=0;bool embedding_probe_loaded_r461=false,embedding_probe_failed_r461=false;','pair_probe.cpp')
 appendix=(D/'pair_append.cpp.inc').read_text(encoding='utf-8');pair+='\n'+appendix
 # This added branch precedes the original RUN; the original complete branch is
 # left byte-for-byte in place. No RankWorker/Runtime/model header fields change.
 worker=replace(worker,'#include "state_only_r9113.hpp"','#include "probe_owned.hpp"\n#include "state_only_r9113.hpp"','worker.cpp')
 needle='            } else if (local_cmd == Command::RUN) {'
 branch='''            } else if (local_cmd == Command::RUN && kimi_embedding_probe_r461::contains(this)) {
                std::shared_ptr<kimi_embedding_probe_r461::Request> request;
                try {
                    request=kimi_embedding_probe_r461::acquire(this,runtime_->device().index());
                    if(!request)throw std::logic_error("r461 admitted request disappeared");
                    kimi_embedding_probe_r461::execute(request,*runtime_,*model_,rank_info_);
                    {std::lock_guard<std::mutex>lock(mutex_);output_=Output{};job_done_=true;}
                    cv_.notify_all();
                }catch(...) {
                    kimi_embedding_probe_r461::cancel_worker(this);
                    fail("embedding_probe_r461",std::current_exception());break;
                }
'''+needle
 worker=replace(worker,needle,branch,'worker.cpp')
 native=D/'native';native.mkdir(exist_ok=True)
 (native/'pair_probe.cpp').write_text(pair,encoding='utf-8',newline='\n')
 (native/'worker.cpp').write_text(worker,encoding='utf-8',newline='\n')
 proof=dict(schema=1,GPU_GO=False,Model_GO=False,CPU_execution_authorized=False,
  source_origins={pair_path:new[pair_path]['SHA256'],worker_path:old[worker_path]['SHA256']},exact_replacements=changes,
  appended_pair_body_SHA256=H(appendix.encode()),new_pair_SHA256=H(pair.encode()),new_worker_SHA256=H(worker.encode()),
  original_nonprobe_RUN_preserved=True,RankWorker_class_header_changed=False,TextModel_class_header_changed=False,
  new_runtime_or_communicator=False,new_GPU_source_or_object=False,
  new_private_Context_fields=['embedding_probe_epoch_r461','embedding_probe_loaded_r461','embedding_probe_failed_r461'])
 (D/'SOURCE_DERIVATION.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return proof
if __name__=='__main__':
 q=assemble(archive(sys.argv[1],'efabab5d822e03c1b340b55aa6c1411ea598f05620b48a8b6649729ebc0d5b40'),
            archive(sys.argv[2],'38e33a3e04592b46448be83cc63b0da06110959be379f8b56258b721d6ed3a02'))
 print(json.dumps({k:v for k,v in q.items()if k!='exact_replacements'},indent=2))
