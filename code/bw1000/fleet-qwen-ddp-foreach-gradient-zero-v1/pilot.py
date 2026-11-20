"""Verify eight exact workers before launching the fleet."""
import json
import os
import socket
import sys
from pathlib import Path

from worker import node_run


def main(path, deadline):
    node=socket.gethostname().split('.')[0]
    parent,step=os.environ['SLURM_JOB_ID'],os.environ['SLURM_STEP_ID']
    if parent!='1103606' or node!='f13r1n11' or not step.isdecimal():
        raise ValueError('PILOT_OWNER')
    work=path.parent/'pilot-workers'/node
    work.parent.mkdir()
    code=node_run(work,dict(node=node,step=parent+'.'+step,launch_epoch=None),deadline)
    terminal=json.loads((work/'terminal.json').read_bytes())
    if code or terminal['error'] is not None or terminal['post_kfd_holders']:
        return 2
    cards=[]
    for i in range(8):
        c=json.loads((work/f'card-{i}.json').read_bytes())
        r,=c['results']
        if r['parameter_tensors']!=504 or r['parameter_elements']!=43646976:
            raise ValueError('PILOT_SHAPES')
        if [b['grouped_zero'] for b in r['caps']] != [False, True, True, False]:
            raise ValueError('PILOT_ABBA')
        for row in r['caps']:
            if (row['gradient_as_bucket_view'] is not True or row['bucket_cap_mib'] != 25
                    or not 0 < row['distinct_live_gradient_storages'] < 504
                    or len(row['timed_steps']) != 2 or len(row['timed_clear_seconds']) != 2
                    or row['actual_grouped_zero_calls'] != (3 if row['grouped_zero'] else 0)
                    or len(row['numeric']) != 2
                    or any(max(step[:2]) > 5e-5 for step in row['numeric'])):
                raise ValueError('PILOT_GRADIENT_RESET_QUALIFICATION')
        cards.append(dict(pci=c['pci'],cap_rows=r['caps']))
    if len({card['pci'] for card in cards}) != 8:
        raise ValueError('PILOT_PCI_BIJECTION')
    with path.open('x',encoding='utf-8') as f:
        json.dump(dict(cards=cards,step=parent+'.'+step,post_clean=True),f,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    return 0


if __name__=='__main__':
    raise SystemExit(main(Path(sys.argv[1]),float(sys.argv[2])))
