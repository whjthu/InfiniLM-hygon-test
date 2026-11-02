"""Create three private HOST TUs from exact retained source bytes; no GPU work."""
from pathlib import Path
import hashlib, json

D = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()

def once(text, old, new):
    assert text.count(old) == 1, ('nonunique patch', old[:100], text.count(old))
    return text.replace(old, new, 1)

def patch(sources, output):
    """sources is the retained remote audit files dictionary (path -> SHA/text)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    repo = D.parent
    pair_key = '/refactor/incoming/Kimi-append512-r247/native/pair_probe.cpp'
    pair = sources[pair_key]['text']
    assert sha(pair.encode()) == sources[pair_key]['SHA256'] == '38dbbe54e90d4150ee510acac774254b9e9e1ff5da36d210e8e36eebf84240ce'
    originals = {'pair_probe.cpp': pair}
    pair = '#include "topology_r410.hpp"\n' + pair
    pair = once(pair, 'thread_local bool authorized_long_create_r179=false;',
        'thread_local bool authorized_long_create_r179=false;\n'
        'thread_local bool authorized_tp8_pp4_r410=false;\n'
        'struct OrdinaryCreateScopeR410 { OrdinaryCreateScopeR410(){'
        'if(authorized_tp8_pp4_r410)throw std::invalid_argument("r410 nested factory");'
        'authorized_tp8_pp4_r410=true;}~OrdinaryCreateScopeR410(){authorized_tp8_pp4_r410=false;} };')
    pair = once(pair, 'struct Context{bool batch_enabled_r9122=false,',
        'struct Context{bool ordinary_tp8_pp4_r410=false;bool batch_enabled_r9122=false,')
    old = 'const bool long_scope_r179=cfg.value("infinilm_kimi_long_agent_r179",false);if(long_scope_r179!=authorized_long_create_r179||(long_scope_r179&&(max_rows!=2||tp_nodes!=2||pp_size!=2)))throw std::invalid_argument("r179 explicit factory scope");auto&text=cfg.at("text_config");text["infinilm_kimi_long_agent_r179"]=long_scope_r179;'
    new = 'const bool long_scope_r179=cfg.value("infinilm_kimi_long_agent_r179",false);'
    new += 'const bool tp8_r410=cfg.value("infinilm_kimi_tp8_pp4_r410",false);'
    new += 'if(tp8_r410!=authorized_tp8_pp4_r410||long_scope_r179!=authorized_long_create_r179||'
    new += '(tp8_r410&&!long_scope_r179)||(long_scope_r179&&(max_rows!=2||'
    new += '!kimi_tp8_pp4_r410::topology(tp8_r410,8*tp_nodes,pp_size))))'
    new += 'throw std::invalid_argument("r410 explicit factory admission");'
    new += 'auto&text=cfg.at("text_config");text["infinilm_kimi_long_agent_r179"]=long_scope_r179;'
    new += 'text["infinilm_kimi_tp8_pp4_r410"]=tp8_r410;'
    pair = once(pair, old, new)
    pair = once(pair, 'auto c=std::make_unique<Context>();c->stage=stage;',
        'auto c=std::make_unique<Context>();c->ordinary_tp8_pp4_r410=tp8_r410;c->stage=stage;')
    # Preserve the original forward implementation. New contexts have a typed C1 entry.
    marker = 'extern "C" int modern_kimi_engine_forward_r2475('
    start = pair.index(marker)
    pos = pair.index('auto*c=static_cast<Context*>(raw);', start)
    pair = pair[:pos] + pair[pos:].replace('auto*c=static_cast<Context*>(raw);',
        'auto*c=static_cast<Context*>(raw);if(c&&c->ordinary_tp8_pp4_r410)'
        'throw std::invalid_argument("r410 typed ordinary entry required");', 1)
    start = pair.index('extern "C" int modern_kimi_engine_resize_cache_r2475(')
    pos = pair.index('auto*c=static_cast<Context*>(raw);', start)
    pair = pair[:pos] + pair[pos:].replace('auto*c=static_cast<Context*>(raw);',
        'auto*c=static_cast<Context*>(raw);if(c&&c->ordinary_tp8_pp4_r410)'
        'throw std::invalid_argument("r410 fixed long capacity");', 1)
    start = pair.index('extern "C" int modern_kimi_engine_set_cache_pages_r4974(')
    pos = pair.index('auto*c=static_cast<Context*>(raw);', start)
    pair = pair[:pos] + pair[pos:].replace('auto*c=static_cast<Context*>(raw);',
        'auto*c=static_cast<Context*>(raw);if(c&&c->ordinary_tp8_pp4_r410)'
        'throw std::invalid_argument("r410 fixed long pages");', 1)
    start = pair.index('extern "C" const char*modern_kimi_long_agent_capabilities_r179(')
    pos = pair.index('auto*c=static_cast<Context*>(raw);', start)
    pair = pair[:pos] + pair[pos:].replace('auto*c=static_cast<Context*>(raw);',
        'auto*c=static_cast<Context*>(raw);if(c&&c->ordinary_tp8_pp4_r410)'
        'throw std::invalid_argument("r410 separate ordinary capabilities");', 1)
    pair += r'''
// No batch/speculation/capture state is constructed by this explicit factory.
extern "C" void*modern_kimi_engine_create_ordinary_tp8_pp4_r410(
 const char*raw,int node,const char*address,int port){try{
 if(!raw||!address||node<0||node>=4)throw std::invalid_argument("r410 factory arguments");
 OrdinaryCreateScopeR410 topology;LongCreateScopeR179 long_scope;
 auto cfg=nlohmann::json::parse(raw);const auto&t=cfg.at("text_config");
 if(t.at("num_attention_heads")!=96||t.at("qk_nope_head_dim")!=128||
 t.at("linear_attn_config").at("num_heads")!=96||t.at("linear_attn_config").at("head_dim")!=128||
 t.value("max_position_embeddings",size_t(0))<32768)
 throw std::invalid_argument("r410 checkpoint geometry");
 cfg["infinilm_kimi_long_agent_r179"]=true;cfg["infinilm_kimi_tp8_pp4_r410"]=true;
 return modern_kimi_engine_create_r2475(cfg.dump().c_str(),node,1,4,address,port,2);
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_kimi_engine_forward_ordinary_r410(void*raw,int kind,
 size_t tokens,size_t count,int flags,const int32_t*rows,const int32_t*past,
 const int32_t*total,const int32_t*offsets,const int64_t*ids,const int64_t*positions,
 void*out_logits,void*out_ids,int*out_flags){try{
 auto*c=static_cast<Context*>(raw);
 if(!c||!c->ordinary_tp8_pp4_r410||!rows||!past||!total||!offsets||
 c->pair_r9006||c->speculation_r3632||c->batch_enabled_r9122||c->state_only_r9114)
 throw std::invalid_argument("r410 ordinary context and metadata");
 kimi_tp8_pp4_r410::call_scope(kind,count,rows[0],past[0],tokens,flags);
 if(int64_t(total[0])!=int64_t(past[0])+int64_t(tokens)||offsets[0]!=0||
 offsets[1]!=int32_t(tokens))throw std::invalid_argument("r410 exact input interval");
 std::lock_guard<std::mutex>lock(c->mutex);
 return input_forward_locked_r247(c,tokens,count,flags,rows,past,total,offsets,
 ids,positions,out_logits,out_ids,out_flags);
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_ordinary_capabilities_r410(void*raw){try{
 auto*c=static_cast<Context*>(raw);if(!c||!c->ordinary_tp8_pp4_r410||
 c->tp_nodes!=1||c->pp_size!=4||c->pair_r9006||c->speculation_r3632)
 throw std::invalid_argument("r410 explicit ordinary context required");
 serialized=nlohmann::json{{"version",1},{"TP",8},{"PP",4},{"C",1},
 {"physical_rows",2},{"state_rows",3},{"pages",260},{"page_size",256},
 {"context",32768},{"prompt_chunk",512},{"decode_width",1},
 {"MLA_shape",{2,260,256,12,192}},{"KDA_heads",12},{"local_MoE_I",384},
 {"speculation",false},{"capture",false},{"GPU_qualified",false}}.dump();
 return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
'''
    rel = Path('agent-context-32k/Kimi-long-agent32K-private-r179/native/kimi_compact_allocate_r4974.cpp')
    allocator_raw = (repo / rel).read_bytes()
    assert sha(allocator_raw) == 'adbfce2c638dcf4fd064eadca33c48418df01840a1aa4fe5211c9e4196f5bdab'
    allocator = allocator_raw.decode('utf-8').replace('\r\n','\n')
    originals['kimi_compact_allocate_r4974.cpp'] = allocator
    allocator = '#include "topology_r410.hpp"\n' + allocator
    allocator = once(allocator,
        'if(compact_capability)infinilm::models::kimi_native::validate_cache(config,true);',
        'const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);\n'
        '    if(compact_capability||long_scope_r179)infinilm::models::kimi_native::validate_cache(config,compact_capability,long_scope_r179);')
    allocator = once(allocator,
        'const bool long_scope_r179=model_config->get_config_json().value("infinilm_kimi_long_agent_r179",false);\n    if(long_scope_r179)',
        'if(long_scope_r179)')
    allocator = once(allocator,
        'if(tp_size!=16||rank_info.pp_size!=2)throw std::invalid_argument("r179 cache TP16PP2 scope");',
        'kimi_tp8_pp4_r410::cache_scope(model_config->get_config_json().value("infinilm_kimi_tp8_pp4_r410",false),tp_size,rank_info.pp_size);')
    scale_dir = repo / 'agent-context-32k/Kimi-long-scale-exact-private-r201'
    moe = (scale_dir / 'kimi_moe_scale_exact_r201.cpp').read_text(encoding='utf-8')
    assert sha(moe.encode()) == '1959878b051f9ad25d60a3f17bb776a9f74623cacccf9fd7d216949addea70eb'
    originals['kimi_moe_scale_exact_r201.cpp'] = moe
    moe = once(moe,
        'runtime.device().type()==infinilm::runtime::Device::Type::kHygon,!runtime.isGraphRecording()},',
        'runtime.device().type()==infinilm::runtime::Device::Type::kHygon,!runtime.isGraphRecording(),\n'
        '             model_config->get_config_json().value("infinilm_kimi_tp8_pp4_r410",false)},')
    scale = (scale_dir / 'scale_exact_scope_r201.hpp').read_text(encoding='utf-8')
    originals['scale_exact_scope_r201.hpp'] = scale
    scale = '#include "topology_r410.hpp"\n' + scale
    scale = once(scale, 'bool hygon,eager;', 'bool hygon,eager;bool ordinary_tp8_pp4=false;')
    scale = once(scale,
        'g.tp!=16||g.pp!=2||g.experts!=896||g.hidden!=3584||g.local_intermediate!=192',
        '!kimi_tp8_pp4_r410::topology(g.ordinary_tp8_pp4,g.tp,g.pp)||g.experts!=896||g.hidden!=3584||g.local_intermediate!=(g.ordinary_tp8_pp4?384u:192u)')
    outputs = {'pair_probe.cpp':pair,'kimi_compact_allocate_r4974.cpp':allocator,
               'kimi_moe_scale_exact_r201.cpp':moe,'scale_exact_scope_r201.hpp':scale,
               'topology_r410.hpp':(D/'topology_r410.hpp').read_text(encoding='utf-8')}
    for name, text in outputs.items():
        (output/name).write_bytes(text.encode())
    receipt={'original_LF_SHA256':{n:sha(t.encode()) for n,t in originals.items()},
             'allocator_original_raw_SHA256':sha(allocator_raw),
             'private_SHA256':{n:sha(t.encode()) for n,t in outputs.items()},
             'new_HOST_TUs':3,'new_GPU_TUs':0,'Model':False,'GPU':False,
             'note':'New ordinary C1 factory; no old batch/state helpers enabled. Actual headers/link/device still pending.'}
    (output/'patch.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('retained_audit');p.add_argument('output')
    a=p.parse_args();print(json.dumps(patch(json.loads(Path(a.retained_audit).read_text(encoding='utf-8-sig'))['files'],a.output),indent=2))
