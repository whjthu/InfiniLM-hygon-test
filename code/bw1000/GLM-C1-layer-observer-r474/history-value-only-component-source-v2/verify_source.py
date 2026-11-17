"""Pure local source structure check only; no imports of binding/Native/SDK."""
import ast,hashlib,json,re
from pathlib import Path
def verify(directory):
    d=Path(directory);s=(d/'component.cpp').read_text(encoding='utf8');old=d.with_name('history-value-only-component-source-v1');parent=(old/'component.cpp').read_text(encoding='utf8')
    h=lambda b:hashlib.sha256(b).hexdigest()
    assert h((old/'component.cpp').read_bytes())=='a917f6b2226f92a2cb335e8fbe19f580ca2575d137a342442be42d974506dfd9'
    abi=lambda text:re.findall(r'extern "C"\s+(?:const char\*|void\*|int)\s*(glm_history_component_[a-z]+)\(',text)
    assert abi(s)==abi(parent) and len(abi(s))==9
    assert s.count('Lease lease(raw);')==6 and 'static_cast<Context*>(raw)' not in s and 'ctx.release()' not in s
    for token in ('r.entries.size()<64&&r.next!=UINT64_MAX','entry->token->generation==entry->generation','entry->owner==std::this_thread::get_id()','entry->context->runtime.get()==entry->runtime_identity','lease.entry->quarantined=true','lease.retire()'):
        assert token in s
    assert 'entries.erase' not in s and 'entries.clear' not in s
    # Exact pre-existing destructor/close and math-call sequences, not a C++ proof.
    for a,b in [(' void close(){',' ~Context()')]:
        assert parent[parent.index(a):parent.index(b)]==s[s.index(a):s.index(b)]
    assert '~Held()noexcept{try{r.syncStream();}catch(...){std::terminate();}}' in s
    assert 'std::vector<std::shared_ptr<std::vector<std::uint8_t>>>host_buffers;' in s
    assert s.index('held.host_buffers.push_back(prefix)')<s.index('void restore(')
    assert s.index('held.host_buffers.push_back(bad)')<s.index('r.memcpyH2D(p,bad->data()')
    assert s.index('held.host_buffers.push_back(a)')<s.index('r.memcpyD2H(a->data()')
    for line in parent.splitlines():
        if any(x in line for x in ('model::history_value_only_v1::selected_values(*c->prepare,r,l,k,index,held.values)','r.syncStream();native_ms=ms(started);','need(v&&v->shape()==rt::Shape{rows,64,256}','auto&out=candidate?b:a;')):assert line in s
    assert 'Redzone latent_guard' in s and 'rotary_guard' in s and 'ids_guard' in s
    assert 'redzone_checks_per_pair",12' in s and 'redzone_detector_planned",6' in s
    for name in ('controlled_inputs.py','CONTROLLED_INPUT_PINS.json','probe_recipe.py'):assert (d/name).read_bytes()==(old/name).read_bytes()
    for p in d.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=p.name)
    bound=json.loads((d/'QUALIFIED_PARAMETER_PINS.json').read_bytes());assert bound['source_SHA256']=='d5ca45e48c613d66d7aafc31cd7b84fb141fc478de94155af807bbc4f92048b1'
    assert len(bound['parameters'])==10 and sum(x['bytes'] for x in bound['parameters'])==64377280 and all(re.fullmatch('[0-9a-f]{64}',x['range_SHA256']) for x in bound['parameters'])
    template=json.loads((d/'PARAMETER_BINDING_TEMPLATE.json').read_bytes());assert template['parameters']==bound['parameters'] and not template['GPU_GO'] and not template['component_CPU_qualified']
    return dict(structural_checks_passed=True,CABI=9,registry_gates=6,handle_validation_planned=24,redzone_detector_validation_planned=6,original_math_call_text_equal=True,original_Context_close_text_equal=True,Held_destructor_body_equal_new_host_owners=True,CPU_compile_run=False,device_run=False)
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent)))
