"""CPU SYMBOLIC CONTRACT ONLY. No native tensors, weights, MTP, or GPU proof.

Opaque integer tags stand for token-dependent projection/cache bytes. Reference
builds state directly from prefix; transaction applies a bounded write set.
"""
import copy
import hashlib
import json
from pathlib import Path

RATIOS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 43)]

def tag(layer, kind, position, token):
    return (layer, kind, position, token)

def selection(layer, position):
    # Deliberately opaque synthetic selection; does NOT implement native topk.
    visible = (position + 1) // 4
    return tuple(range(max(0, visible - 512), visible))

def direct_layer(layer, tokens):
    n, ratio = len(tokens), RATIOS[layer]
    ring = {p % 128: tag(layer, 'kv', p, tokens[p]) for p in range(max(0,n-128),n)}
    state, indexstate, compressed, keys = {}, {}, {}, {}
    if ratio:
        width = 8 if ratio == 4 else 128
        for p in range(max(0,n-width),n):
            state[p % width] = tag(layer, 'score512', p, tokens[p])
            if ratio == 4:
                indexstate[p % width] = tag(layer, 'score128', p, tokens[p])
        for g in range(n // ratio):
            end = (g+1)*ratio
            start = max(0,end-(8 if ratio == 4 and end >= 8 else ratio))
            compressed[g] = tuple(tag(layer,'score512',p,tokens[p]) for p in range(start,end))
            if ratio == 4:
                keys[g] = tuple(tag(layer,'score128',p,tokens[p]) for p in range(start,end))
    return dict(length=n, ring=ring, state=state, indexstate=indexstate,
                compressed=compressed, keys=keys,
                last_selection=selection(layer,n-1) if ratio==4 and n else ())

def direct(tokens):
    return [direct_layer(layer,tokens) for layer in range(43)]

class SymbolicModel:
    def __init__(self, token_rows):
        self.owner = object()
        self.epoch = 0
        self.generation = [0]*len(token_rows)
        self.rows = [direct(tokens) for tokens in token_rows]

    def prepare(self, proposals):
        # projections are two opaque position-tagged values per affected row.
        return dict(owner=self.owner,epoch=self.epoch,generation=self.generation[:],
                    proposals=copy.deepcopy(proposals), used=False)

    def commit(self, pending, counts, fail_layer=None):
        if pending['used'] or pending['owner'] is not self.owner or pending['epoch']!=self.epoch or pending['generation']!=self.generation:
            raise ValueError('stale/foreign/consumed')
        if set(counts)!=set(pending['proposals']):
            raise ValueError('row cohort differs')
        for row,count in counts.items():
            if not 0<=count<=len(pending['proposals'][row])<=2:
                raise ValueError('invalid prefix')
        shadow=copy.deepcopy(self.rows)
        for layer,ratio in enumerate(RATIOS):
            if layer==fail_layer:
                raise RuntimeError('injected prepublication failure')
            for row,count in counts.items():
                s=shadow[row][layer]
                past=s['length']
                for offset,token in enumerate(pending['proposals'][row][:count]):
                    p=past+offset
                    s['ring'][p%128]=tag(layer,'kv',p,token)
                    if ratio:
                        width=8 if ratio==4 else 128
                        s['state'][p%width]=tag(layer,'score512',p,token)
                        if ratio==4:
                            s['indexstate'][p%width]=tag(layer,'score128',p,token)
                        if (p+1)%ratio==0:
                            start=p+1-(8 if ratio==4 and p>=7 else ratio)
                            s['compressed'][p//ratio]=tuple(s['state'][q%width] for q in range(start,p+1))
                            if ratio==4:
                                s['keys'][p//ratio]=tuple(s['indexstate'][q%width] for q in range(start,p+1))
                        if ratio==4:
                            s['last_selection']=selection(layer,p)
                    s['length']=p+1
        # One host reference publication models all-layer atomic visibility.
        # Native multi-owner device atomic publication is NOT implemented here.
        self.rows=shadow
        self.epoch+=1
        pending['used']=True

    def abort(self,pending):
        if pending['owner'] is not self.owner or pending['epoch']!=self.epoch or pending['used']:
            raise ValueError('stale/foreign/consumed')
        self.epoch+=1
        pending['used']=True

    def reset(self,row):
        self.rows[row]=direct([])
        self.generation[row]+=1
        self.epoch+=1

def must_reject(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('invalid transaction accepted')

def run():
    cases=[]
    boundaries=[0,1,2,3,4,5,6,7,8,126,127,128,129,130,2046,2047,2048,2049]
    for past in boundaries:
        base=list(range(past))
        for count in (0,1,2):
            m=SymbolicModel([base,[41,42,43]])
            old=copy.deepcopy(m.rows)
            p=m.prepare({0:[90001,90002]})
            m.commit(p,{0:count})
            assert m.rows[0]==direct(base+[90001,90002][:count])
            assert m.rows[1]==old[1]
            must_reject(lambda:m.commit(p,{0:count}))
            cases.append(dict(past=past,commit_inputs=count,acceptance='abort' if count==0 else count-1))
    # Mixed ragged cohort, accept0 drafts=>one anchor, accept1=>two inputs.
    m=SymbolicModel([list(range(127)),list(range(3))])
    p=m.prepare({0:[70,71],1:[80,81]})
    m.commit(p,{0:1,1:2})
    assert m.rows==[direct(list(range(127))+[70]),direct(list(range(3))+[80,81])]
    # Rejected EOS must not end request; accepted EOS keeps only causal prefix.
    eos=1
    for proposal,accepted,count,emitted,terminal in [([90,eos],False,1,[91],False),([90,eos],True,2,[eos],True),([90,92],False,1,[eos],True)]:
        m=SymbolicModel([list(range(127))]);p=m.prepare({0:proposal});m.commit(p,{0:count})
        assert m.rows[0]==direct(list(range(127))+proposal[:count])
        assert terminal==(eos in emitted)
    # Known terminal anchor: no verification/draft; zero target/MTP calls.
    terminal_anchor=dict(anchor=eos,target_calls=0,mtp_calls=0,verify_inputs=0)
    assert terminal_anchor['verify_inputs']==0
    # Cancellation/abort cannot leak tentative slots or revisions.
    m=SymbolicModel([list(range(129))]);old=copy.deepcopy(m.rows)
    p=m.prepare({0:[70,71]});m.abort(p);assert m.rows==old
    must_reject(lambda:m.commit(p,{0:1}))
    p=m.prepare({0:[70,71]});m.reset(0)
    must_reject(lambda:m.commit(p,{0:1}));assert m.rows==[direct([])]
    other=SymbolicModel([[]]);foreign=other.prepare({0:[70,71]})
    must_reject(lambda:m.commit(foreign,{0:1}))
    for layer in (0,1,21,42):
        m=SymbolicModel([list(range(127))]);p=m.prepare({0:[70,71]});old=copy.deepcopy(m.rows)
        try:m.commit(p,{0:2},fail_layer=layer)
        except RuntimeError:pass
        else:raise AssertionError('injected fault ignored')
        assert m.rows==old
    # Deliberately demonstrate why merely shortening lengths is incorrect.
    m=SymbolicModel([list(range(127))]);p=m.prepare({0:[70,71]});m.commit(p,{0:2})
    for s in m.rows[0]:s['length']=128
    assert m.rows[0]!=direct(list(range(127))+[70])
    return dict(status='PASS_SYMBOLIC_ONLY_NOT_NATIVE',prefix_cases=len(cases),
                layer_prefix_comparisons=len(cases)*43,case_matrix=cases,
                additional=['ragged mixed accept0/1','EOS accepted/rejected/replacement','terminal anchor short circuit',
                'cancel abort','row reset generation','foreign pending','duplicate consumed','failure before publish at layers0/1/21/42',
                'counterexample length-only truncate'],
                limits=['Opaque tags do not compute attention, compressor pooling, BF16 rounding or native topk.',
                        'CPU object publication does not prove eight-owner GPU failure atomicity.',
                        'Reference prefix construction is test-only and is not proposed runtime history replay.'])

if __name__=='__main__':
    result=run()
    dst=Path(__file__).with_name('deep_prefix_oracle_result_r118.json')
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='case_matrix'},ensure_ascii=False))
