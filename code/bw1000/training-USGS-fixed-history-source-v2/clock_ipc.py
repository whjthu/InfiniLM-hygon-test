"""Binary-float Job clock IPC only; scientific GeoJSON parser is unchanged."""
import json
import math


def validate_gate(raw,epoch,deadline,actual_mask):
    def need(ok):
        if not ok:raise ValueError('Actual Job/epoch differs')
    def pairs(items):
        need(len(items)==len({k for k,v in items}));return dict(items)
    def constant(value):raise ValueError('Nonfinite IPC JSON')
    gate=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    need(isinstance(gate,dict))
    clock=gate.get('deadline')
    need(type(epoch) in (int,float) and type(deadline) in (int,float) and type(clock) in (int,float))
    need(all(math.isfinite(x) for x in [epoch,deadline,clock]) and deadline-epoch==120 and clock==deadline)
    need(type(gate.get('CPU_count')) is int and gate['CPU_count']==2
        and type(gate.get('memory_limit_bytes')) is int and gate['memory_limit_bytes']==2147483648
        and gate.get('job_affinity_verified') is True
        and type(gate.get('worker_affinity_mask')) is int and gate['worker_affinity_mask']>0
        and type(actual_mask) is int and actual_mask==gate['worker_affinity_mask'])
    return gate
