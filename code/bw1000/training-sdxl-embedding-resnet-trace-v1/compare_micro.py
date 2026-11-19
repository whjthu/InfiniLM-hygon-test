"""Compare the original SDXL time and first ResNet traces without tensor data."""
def compare(ref, hip):
    for field in ('events','call_counts','first_resnet_config'):
        if ref[field]!=hip[field]:
            raise ValueError('TRACE_TOPOLOGY_CHANGED: '+field)
    if set(ref['edges'])!=set(hip['edges']) or len(ref['events'])!=len(set(ref['events'])):
        raise ValueError('TRACE_EDGE_SET_CHANGED')
    changes=[]
    for key in ref['events']:
        a,b=(x['edges'][key] for x in (ref,hip))
        if (a['shape'],a['dtype'])!=(b['shape'],b['dtype']):
            raise ValueError('TRACE_TYPE_CHANGED: '+key)
        if a['sha256']!=b['sha256']:
            changes.append(key)
    return dict(earliest=changes[0] if changes else None,
                matches=[x for x in ref['events'] if x not in changes],mismatches=changes)
