"""New tail20 input/forward/RNG/loss checks only; no model/complete-run fixture."""
import ast
import copy
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    tree = ast.parse((D/'read_original.py').read_bytes())
    funcs = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('need','tail_matches')]
    scope = dict(math=math,PAIR='d58ebcf4dc0345b2a3719067f0fe6961')
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual-tail-match','exec'),scope)
    old = dict(mode='fresh25',arm='domain',pair_id=scope['PAIR'],training_inputs=[],forward_inputs=[],losses=[])
    for i in range(100):
        old['training_inputs'].append(dict(microbatch=i,global_step_before=i//4,canonical_ordinal=i%45,
             source_id_SHA256='a'*64,collated_fields_SHA256='b'*64,training_inputs_SHA256='c'*64,
             input_tokens=400,effective_supervised_tokens=10,supervised_EOS_count=1))
        old['forward_inputs'].append(dict(microbatch=i,input_SHA256='d'*64,RNG_before=dict(finite=True,SHA256='e'*64)))
        old['losses'].append(.125+i/1024)
    new = copy.deepcopy(old);new.update(mode='resume20to25',training_inputs=copy.deepcopy(old['training_inputs'][80:]),
        forward_inputs=copy.deepcopy(old['forward_inputs'][80:]),losses=old['losses'][80:],actual_input_tokens=8000,actual_supervised_tokens=200)
    for i in range(20):new['training_inputs'][i]['microbatch']=new['forward_inputs'][i]['microbatch']=i
    assert scope['tail_matches'](new,old,0)['input_forward_RNG_loss_tail_exact'] is True
    rejected=[]
    for label,change in [('loss-drift',lambda x:x['losses'].__setitem__(0,.999)),
                         ('forward-RNG-drift',lambda x:x['forward_inputs'][0]['RNG_before'].update(SHA256='0'*64))]:
        bad=copy.deepcopy(new);change(bad)
        try:scope['tail_matches'](bad,old,0)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    print(json.dumps(dict(passed=True,valid_tail20=1,rejected=rejected,remote_calls=0,Torch_GPU_calls=0,
                          fabricated_full_run_fixture=False)))


if __name__=='__main__':main()
