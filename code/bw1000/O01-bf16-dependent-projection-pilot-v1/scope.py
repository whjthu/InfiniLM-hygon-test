"""One independent O01 node from the fixed 80-node ledger."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re

from contract import FIRST_ACTIVE, WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS, assign_cohorts, grid


LABEL = 'O01-BF16-dependent-projection-pilot-v1'
REQUEST = '4bde1b3c09863a190bf3988881b0d73b'
RUN = LABEL + '-' + REQUEST
PARENTS = ('1087656','1103606')
ZONES = ('O01',*(f'I{i:02d}' for i in range(1,4)),*(f'T{i:02d}' for i in range(1,13)))
LEDGER_SHA = '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'
PILOT_SHA = '529479e4b108f65947257d5efa0a9a3288c102a2def7492bd559bce117ac3fd2'
EXCLUDED_NODES = frozenset(('f11r2n15','f11r4n09','f11r4n10','f11r4n19','f12r2n12'))
PILOT_NODE = "f12r2n13"
PILOT_PARENT = "1103606"
PILOT_LABEL = "O01-bf16-dependent-"+REQUEST[:12]
PCI = re.compile(r'[a-f0-9]{4}:[a-f0-9]{2}:[01][a-f0-9]\.[0-7]')
HEX = re.compile(r'[a-f0-9]{64}')


def need(condition, why):
    if not condition:
        raise ValueError(why)


def ledger(raw):
    need(sha256(raw).hexdigest()==LEDGER_SHA,'actual owned ledger SHA differs')
    decoded=json.loads(raw)
    owners={}
    for zone,entry in decoded['groups'].items():
        need(zone in ZONES,'unfrozen resource zone')
        for parent,nodes in entry['allocations'].items():
            need(parent in PARENTS,'unfrozen parent')
            for node in nodes:
                need(node not in owners,'duplicate ledger node')
                owners[node]=(parent,zone)
    return owners


def staged_ledger():
    return ledger(Path(__file__).resolve().with_name('ledger.json').read_bytes())


def placement(selected,owners):
    need(type(selected) is dict and set(selected)=={PILOT_NODE}
         and not EXCLUDED_NODES.intersection(selected),'pilot requires its one exact owned O01 node')
    item=selected[PILOT_NODE]
    pci,before=item['PCI'],item['before_CPU_step']
    need((item['parent'],item['zone'])==owners[PILOT_NODE]==(PILOT_PARENT,'O01')
         and type(pci) is list and len(pci)==len(set(pci))==8
         and all(isinstance(device,str) and PCI.fullmatch(device) for device in pci)
         and isinstance(before,str) and re.fullmatch(PILOT_PARENT+r'\.[0-9]+',before),
         'original O01 ledger parent, PCI or CPU prerequisite differs')
    return {PILOT_LABEL:dict(parent=PILOT_PARENT,zone='O01',nodes=[PILOT_NODE],
                              request=sha256((REQUEST+':'+PILOT_LABEL).encode()).hexdigest()[:32])}


def validate_plan(plan,owners):
    epoch=plan['campaign_epoch']
    grid(plan,0)
    need(plan['request']==REQUEST and type(epoch) is float and math.isfinite(epoch)
         and plan['shared_start_epoch']==epoch+FIRST_ACTIVE[0] and plan['deadline_epoch']==epoch+WORKER_SECONDS
         and plan['pilot_SHA256']==PILOT_SHA
         and all(HEX.fullmatch(plan[key]) for key in ('source_SHA256','before_snapshot_SHA256'))
         and re.fullmatch(r'[a-f0-9]{40}',plan['publication']),'original campaign or fixed absolute epoch differs')
    shards=placement(plan['selected'],owners)
    need(shards==plan['shards'] and assign_cohorts(shards)==plan['cohort_by_shard'],
         'derived fixed O01 parent or pilot membership differs')
    return shards


def validate_worker_plan(plan,node,parent,step):
    shards=validate_plan(plan,staged_ledger())
    need(step.isdecimal() and any(unit['parent']==parent and node in unit['nodes'] for unit in shards.values()),
         'Slurm task must be an admitted node in its numbered parent')
    return plan['selected'][node]
