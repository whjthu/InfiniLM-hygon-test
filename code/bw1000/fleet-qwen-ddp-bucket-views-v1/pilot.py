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
        if [b['gradient_as_bucket_view'] for b in r['caps']]!=[False,True,True,False]:
            raise ValueError('PILOT_ABBA')
        cards.append(dict(pci=c['pci'],cap_rows=r['caps']))
    with path.open('x',encoding='utf-8') as f:
        json.dump(dict(cards=cards,step=parent+'.'+step,post_clean=True),f,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    return 0


if __name__=='__main__':
    raise SystemExit(main(Path(sys.argv[1]),float(sys.argv[2])))
