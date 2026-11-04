"""Extract exact existing production Weight/Embedding bodies for recording leaf."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
def make(source_paths,output):
    texts={}
    for key,pin in source_paths.items():
        b=Path(pin['path']).read_bytes();assert H(b)==pin['SHA256'];texts[key]=b.decode()
    weight_h=texts['weights.hpp'];weight_h=weight_h[weight_h.index('namespace infinilm::weights'):]
    # One fixture-only access shim, no production fields or body changed.
    needle='    void load_no_sync(Runtime &runtime, const Tensor &source);'
    assert weight_h.count(needle)==1
    weight_h=weight_h.replace(needle,needle+'\n    void recording_load_batch(Runtime&r,const Tensor&s,LoadBatch&b){load_into(r,s,&b);}')
    weight=texts['weights.cpp'];weight=weight[weight.index('namespace infinilm::weights'):weight.index('void ModelWeights::collect')]+ '\n}\n'
    embedding_h=texts['embedding.hpp'];embedding_h=embedding_h[embedding_h.index('namespace infinilm::layers'):]
    embedding=texts['embedding.cpp'];embedding=embedding[embedding.index('namespace infinilm::layers'):embedding.index('Tensor Embedding::forward')]+ '\n}\n'
    private=(D/'vocab_parallel_embedding_r461.hpp').read_text()
    binding=private[private.index('        auto backing=weight_.tensor();'):private.index('        auto &v=vocab_r461::census')]
    # Compile the actual new rebind/alias/shape predicates in the fixture subclass.
    derived='''namespace infinilm::layers {
class RecordingParallel final:public Embedding {
    std::size_t rank_;
public:
    RecordingParallel(runtime::Runtime&r,std::size_t rank):Embedding(r,5120,7168,std::nullopt,runtime::DataType::kBFloat16),rank_(rank){
'''+binding+'''    }
    weights::Weight registered()const{return local_weights_.at("weight");}
};}
'''
    capi=(D/'native/pair_probe.cpp').read_text()
    guard=next(line for line in capi.splitlines()if line.startswith('if(c->vocab_embedding_r461)kimi_vocab_r461::validate_global_ids'))
    boundary='''struct RecordingContextR461 {bool vocab_embedding_r461;};
inline void recording_native_input_boundary(RecordingContextR461*c,const std::int64_t*ids,std::size_t tokens,std::size_t&submitted){
'''+guard+'''\n++submitted;
}
'''
    header='#pragma once\n#include "CPU_tensor_leaf.hpp"\n'+weight_h+'\n'+weight+'\n'+embedding_h+'\n'+embedding+'\n'+derived+'\n'+boundary
    output=Path(output);output.write_text(header,encoding='utf8',newline='\n')
    return dict(source_SHA256={key:pin['SHA256']for key,pin in source_paths.items()},
        actual_weight_bodies_SHA256=H(weight.encode()),actual_embedding_constructor_SHA256=H(embedding.encode()),
        actual_new_binding_SHA256=H(binding.encode()),actual_Native_CPU_ID_guard_SHA256=H(guard.encode()),generated_SHA256=H(output.read_bytes()),
        fixture_only_shim='Weight::recording_load_batch forwards to exact original private load_into; no production change',
        leaf='Recording Tensor/Runtime address, ownership and copy boundaries, not actual SDK/GPU execution')
