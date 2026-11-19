"""Locate mismatch boundaries from summaries that contain no private tensor values."""
import argparse
import json
from pathlib import Path


def compare(a, b):
    if (a['events'] != b['events'] or set(a['edges']) != set(b['edges'])
            or a['target_count'] != b['target_count']):
        raise ValueError('BOUNDARY_GRAPH_CHANGED')
    edges_a, edges_b = a['edges'], b['edges']
    event_order = []
    for event in a['events']:
        name, phase = event.rsplit(':', 1)
        keys = (name+'.in',name+'.residual_in.') if phase=='enter' else (name+'.out',name+'.residual_out.')
        event_order.extend(k for k in edges_a if k==keys[0] or k.startswith(keys[1]))
    if len(event_order)!=len(set(event_order)) or set(event_order)!=set(edges_a):
        raise ValueError('TRACE_EVENT_EDGE_BIJECTION')
    def alias_partition(edges):
        groups = {}
        for name, row in edges.items():
            groups.setdefault(row['alias_id'],set()).add(name)
        return {frozenset(names) for names in groups.values()}
    if alias_partition(edges_a)!=alias_partition(edges_b):
        raise ValueError('TENSOR_ALIAS_GRAPH_CHANGED')
    values, adjoints, unknown = [], [], []
    adjoint_order = {}
    for edge in event_order:
        left,right=edges_a[edge],edges_b[edge]
        for item in ('value','aggregate_adjoint'):
            x,y=left[item],right[item]
            if x is None or y is None:
                if left['adjoint_status']!='not_required' or right['adjoint_status']!='not_required':
                    unknown.append(edge)
                continue
            if (x['shape'],x['dtype'])!=(y['shape'],y['dtype']):
                raise ValueError('SAME_INPUT_TRACE_SCHEMA_MISMATCH')
            if x['sha256']!=y['sha256']:
                (values if item=='value' else adjoints).append(edge)
        adjoint_order.setdefault(left['alias_id'],[]).append(edge)
    mismatched=set(adjoints)
    ordered_adjoint=[names for n in a['adjoint_events'] if (names:=adjoint_order[n])
                     and any(k in mismatched for k in names)]
    return dict(earliest_forward_value=values[0] if values else None,
                forward_value_mismatch_edges=values,
                first_aggregate_adjoint_alias_in_A_reference_hook_order=ordered_adjoint[0] if ordered_adjoint else None,
                aggregate_adjoint_mismatch_edges=adjoints, unknown_adjoint_edges=unknown,
                topology_matches=True, aggregate_adjoint_scope=a['scope'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('reference',type=Path)
    p.add_argument('candidate',type=Path)
    args=p.parse_args()
    print(json.dumps(compare(*(json.loads(x.read_bytes()) for x in (args.reference,args.candidate))),
                     indent=2,allow_nan=False))


if __name__=='__main__':
    main()
