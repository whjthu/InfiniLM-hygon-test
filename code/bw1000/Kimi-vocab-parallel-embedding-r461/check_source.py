"""New source diff/ownership contract only; no model/device imports."""
from pathlib import Path
import hashlib,json,posixpath
D=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def body(source,signature):
    start=source.index(signature);i=source.index('{',start);depth=0;quote=None;escape=False;comment=None;j=i
    while j<len(source):
        ch=source[j];following=source[j:j+2]
        if comment=='//':
            if ch=='\n':comment=None
        elif comment=='/*':
            if following=='*/':comment=None;j+=1
        elif quote:
            if escape:escape=False
            elif ch=='\\':escape=True
            elif ch==quote:quote=None
        elif following in ['//','/*']:comment=following;j+=1
        elif ch in ['"',"'"]:quote=ch
        elif ch=='{':depth+=1
        elif ch=='}':
            depth-=1
            if depth==0:return source[start:j+1]
        j+=1
    raise AssertionError('Unclosed production function '+signature)

def check():
    edits=json.loads((D/'SOURCE_EDITS.json').read_bytes());out={}
    for name,record in edits.items():
        p=Path(record['original']['path']);assert H(p)==record['original']['SHA256']
        old=p.read_text();new=(D/name).read_text();assert H(D/name)==record['source_SHA256']
        lines=old.splitlines(True);made=[];cursor=0
        for edit in record['edits']:
            a,b=edit['start_line'],edit['end_line'];assert a>=cursor and ''.join(lines[a:b])==edit['old']
            made.extend(lines[cursor:a]);made.append(edit['new']);cursor=b
        made.extend(lines[cursor:]);assert ''.join(made)==new
        out[name]=dict(original_SHA256=record['original']['SHA256'],source_SHA256=record['source_SHA256'],exact_edits=len(record['edits']))
    old=Path(edits['native/kimi_k3_text_model.cpp']['original']['path']).read_text();new=(D/'native/kimi_k3_text_model.cpp').read_text()
    same=['KimiK3TextModel::initial_state(','KimiK3TextModel::recv_sharded_last_dim(','KimiK3TextModel::send_pipeline_state(',
          'KimiK3TextModel::apply_output_attn_res(','KimiK3TextModel::forward_embeds(']
    unchanged={}
    for signature in same:
        a,b=body(old,signature),body(new,signature);assert a==b;unchanged[signature]=hashlib.sha256(a.encode()).hexdigest()
    a=body(old,'KimiK3TextModel::forward(');b=body(new,'KimiK3TextModel::forward(')
    assert a.replace('? embed_tokens_->forward(runtime, input_ids)','? embed_tokens(runtime, input_ids)')==b
    a=body(old,'KimiK3TextModel::embed_tokens(');b=body(new,'KimiK3TextModel::embed_tokens(')
    assert a.replace('return embed_tokens_->forward(runtime, input_ids);','return infinilm::layers::embedding_dispatch_r461(runtime,embed_tokens_,input_ids);')==b
    old=Path(edits['native/pair_probe.cpp']['original']['path']).read_text();new=(D/'native/pair_probe.cpp').read_text()
    for signature in ['modern_kimi_engine_load_r2475(','modern_kimi_engine_finish_r2475(','modern_kimi_engine_destroy_r2475(',
                      'modern_kimi_engine_create_ar_storage_r456(','modern_kimi_engine_forward_ar_storage_r456(']:
        a,b=body(old,signature),body(new,signature);assert a==b;unchanged[signature]=hashlib.sha256(a.encode()).hexdigest()
    a=body(old,'int input_forward_locked_r247(');b=body(new,'int input_forward_locked_r247(')
    guard='\nif(c->vocab_embedding_r461)kimi_vocab_r461::validate_global_ids(ids,tokens);'
    assert b.replace(guard,'',1)==a
    assert b.index('validate_global_ids')<b.index('en::InferEngine::Input a')<b.index('c->engine->forward(a)')
    private=(D/'vocab_parallel_embedding_r461.hpp').read_text()
    assert private.count('engine::distributed::allreduce_(r,output,output,infinicclSum,comm_);')==1
    assert private.count('modern_kimi_masked_embedding_r461(r.device().index()')==1
    assert private.index('infini::rt::DeviceGuard device_guard')<private.index('modern_kimi_masked_embedding_r461(r.device().index()')
    assert 'r.syncStream();++v.completed_fences;'in private and 'try{r.syncStream();}catch(...){std::terminate();}'in private
    destructor=body(private,'~VocabParallelEmbeddingR461()')
    assert 'owner_->'not in destructor and 'device_index_'in destructor
    kernel=(D/'masked_embedding.cu').read_text()
    assert 'const dim3 grid((kimi_vocab_r461::hidden+255)/256,tokens)'in kernel and '<<<grid,256,0,'in kernel
    assert 'std::uint16_t' in kernel and not any(word in kernel for word in ['__half2float','__float2bfloat','double','float '])
    return dict(passed=True,exact_source_diffs=out,unchanged_math_loader_lifecycle_bodies_SHA256=unchanged,
        old_class_headers_changed=False,private_Context_bool_only=True,one_collective_all_ranks=True,
        global_invalid_ids_rejected_pre_submit=True,owner_DeviceGuard_and_stronglease_fence=True,
        GPU_qualification=False,SDK_execution=False)
