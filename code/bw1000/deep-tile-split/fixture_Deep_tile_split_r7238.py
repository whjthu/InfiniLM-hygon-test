"""CPU orchestration of the changed flags, not model numeric validation."""
import hashlib
import json
import pathlib
import tempfile
import types
import numpy as np

r = pathlib.Path(__file__).resolve().parent
source = (r/'Deep_causal_teacher_matrix_r7238.py').read_text(encoding='utf-8')
with tempfile.TemporaryDirectory(prefix='deep-tile-split-') as td:
    p = pathlib.Path(td)
    (p/'results').mkdir()
    frozen = [[i % 5 + 1 for i in range(512)] for _ in range(8)]
    ref = p/'results/deep-attention-r3509-910406.102.json'
    ref.write_text(json.dumps({'attention_F64_cases': [{'phase': 'natural-l0-r1', 'generated_ids': frozen}], 'natural_input_provenance': {'cases': [{'input_ids': [1]*512}, {'input_ids': [2]*512}]}}))
    (p/'results/deep-serving-cases-r2130-fixture.json').write_text(json.dumps({'cases': [{'input_ids': [1]*512}, {'input_ids': [2]*512}]}))
    ns = {}
    exec(compile(source.replace('91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9', hashlib.sha256(ref.read_bytes()).hexdigest()), '<explicit-CPU-fixture>', 'exec'), ns)
    state = {'lengths': {}, 'modes': [0,0,0], 'calls': [], 'resets': 0}
    class F:
        def __init__(self, fn): self.fn = fn
        def __call__(self, *args): return self.fn(*args)
    def reset(ctx):
        state['lengths'] = {}
        state['resets'] += 1
        return 0
    lib = types.SimpleNamespace(modern_native_engine_reset=F(reset))
    for name, i in [('modern_deep_layout_policy_r7052',0), ('deep_hc_canonical_policy_r7130',1), ('deep_moe_tail_set_r7211',2)]:
        setattr(lib, name, F(lambda v, i=i: state['modes'].__setitem__(i,v) or 0))
    lib.modern_deep_precision_policy_r3443 = F(lambda v: 0)
    def forward(ids, positions, rows, past, total, offsets):
        steps = []
        for j, (row, old, new) in enumerate(zip(rows,past,total)):
            assert state['lengths'].get(row,0) == old
            step = 0 if old == 0 else old-512+1
            inp = ids[offsets[j]:offsets[j+1]]
            assert inp == ([1 if row % 2 == 0 else 2]*512 if step == 0 else [frozen[row % 8][step-1]])
            assert new == old + len(inp)
            state['lengths'][row] = new
            steps.append(step)
        state['calls'].append({'reset': state['resets'], 'rows': rows.copy(), 'steps': steps, 'modes': state['modes'].copy(), 'tiles': [g['projection_tile_mode'],g['c4_tile_mode']]})
        values = np.zeros((len(rows),16),np.float32)
        for i,s in enumerate(steps): values[i,frozen[0][s]] = 2
        bits = (values.view(np.uint32)>>16).astype(np.uint16)
        return 0,bits,values.argmax(-1).astype(np.int64)
    def check(code): assert code == 0
    g = {'r':p,'tag':'fixture','lib':lib,'ctx':1,'cfg':{'vocab_size':16},'forward':forward,'check':check,'report':{},'path':p/'report.json','torch':types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda d:(10*1024**3,64*1024**3)))}
    ns['run'](g)
    calls = state['calls']
    assert len(calls) == 129 and state['resets'] == 4
    assert len(g['report']['causal_teacher_r7238']) == 3
    for n,tiles in [(1,[16,16]),(2,[0,16]),(3,[16,0])]:
        pre = [c for c in calls if c['reset']==n and c['steps'][0]==0]
        dec = [c for c in calls if c['reset']==n and c['steps'][0]>0]
        assert len(pre)==8 and all(c['tiles']==[0,0] and len(c['rows'])==4 for c in pre)
        assert len(dec)==35 and all(c['tiles']==tiles and c['rows']==list(range(32)) for c in dec)
    assert all(c['modes']==[0,0,0] for c in calls)
    # A bad control must stop before either single-factor arm is entered.
    bad = source.replace('token==frozen[0][step]', 'False')
    bad_ns = {}
    exec(compile(bad.replace('91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9',hashlib.sha256(ref.read_bytes()).hexdigest()),'<bad-control-fixture>','exec'),bad_ns)
    for raw in (p/'results').glob('deep-causal-teacher-r7238-*.bin'): raw.unlink()
    state.update(lengths={},calls=[],resets=0)
    g['report']={}
    try:
        bad_ns['run'](g)
    except AssertionError as exc:
        assert 'A_control_must_reproduce' in str(exc)
        assert state['resets']==1
    else:
        raise AssertionError('bad control was not rejected')
    result = {'passed':True,'three_single_factor_arms':True,'identical_prefill_and_decode_geometry':True,'frozen_teacher_allsteps':True,'forward_calls':129,'control_failure_stops_before_diagnostic_arms':True,'production_matrix_SHA256':hashlib.sha256(source.encode()).hexdigest(),'scope':'Synthetic CPU orchestration; only fixture reference SHA replaced in memory, no DSO/GPU numeric evidence.'}
    output = r/'Deep-tile-split-CPU-r7238.json'
    assert not output.exists()
    output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result))
