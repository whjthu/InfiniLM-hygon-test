"""Ledger-bound dependency and pinned copy interaction on eligible nodes."""

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from contract import FIRST_ACTIVE, WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS, assign_cohorts, grid


LABEL = 'O01-BF16-dependent-d2h-chunks-fleet-v1'
REQUEST = '93c5218eea4e4f1d8a246ba7583dac85'
RUN = LABEL + '-' + REQUEST
PARENTS = ('1087656','1103606')
ZONES = ('O01',*(f'I{i:02d}' for i in range(1,4)),*(f'T{i:02d}' for i in range(1,13)))
LEDGER_SHA = '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'
PILOT_SHA = '210a2e4f5a7c5036547b85505c895f36a52d1c9e75f48544f22b9d7069bf2d90'
EXCLUDED_NODES = frozenset(('f11r2n15','f11r4n09','f11r4n10','f11r4n19','f12r2n12'))
PRIORITY_NODES = frozenset(('f13r1n12',))
MAX_NODES = 74
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
    need(type(selected) is dict and 0<len(selected)<=len(owners),'empty or impossible batch selection')
    groups=defaultdict(list)
    for node,item in sorted(selected.items()):
        parent,zone=item.get('parent'),item.get('zone')
        pci=item.get('PCI')
        before=item.get('before_CPU_step')
        need(node in owners and (parent,zone)==owners[node] and type(pci) is list
             and len(pci)==len(set(pci))==8 and all(isinstance(device,str) and PCI.fullmatch(device) for device in pci)
             and isinstance(before,str) and re.fullmatch(parent+r'\.[0-9]+',before),'CPU-only step, PCI or zone/parent differs')
        groups[parent,zone].append(node)
    output={}
    for (parent,zone),members in sorted(groups.items()):
        for offset in range(0,len(members),4):
            label=f'O01-dependent-d2h-chunks-{REQUEST[:12]}-{parent}-{zone}-{offset//4:02d}'
            output[label]=dict(parent=parent,zone=zone,nodes=members[offset:offset+4],
                               request=sha256((REQUEST+':'+label).encode()).hexdigest()[:32])
    return output


def validate_plan(plan,owners):
    epoch=plan['campaign_epoch']
    grid(plan,0)
    need(plan['request']==REQUEST and type(epoch) is float and math.isfinite(epoch)
         and plan['shared_start_epoch']==epoch+FIRST_ACTIVE[0] and plan['deadline_epoch']==epoch+WORKER_SECONDS
         and plan['pilot_SHA256']==PILOT_SHA
         and all(HEX.fullmatch(plan[key]) for key in ('source_SHA256','before_snapshot_SHA256'))
         and re.fullmatch(r'[a-f0-9]{40}',plan['publication']),'original campaign or fixed absolute epoch differs')
    need(0 < len(plan['selected']) <= MAX_NODES and not (EXCLUDED_NODES|PRIORITY_NODES).intersection(plan['selected']),
         'selection exceeds the 74-node scope or includes a protected node')
    shards=placement(plan['selected'],owners)
    need(shards==plan['shards'] and assign_cohorts(shards)==plan['cohort_by_shard'],
         'derived group/parent or fixed staggered cohort membership differs')
    return shards


def validate_worker_plan(plan,node,parent,step):
    shards=validate_plan(plan,staged_ledger())
    need(step.isdecimal() and any(unit['parent']==parent and node in unit['nodes'] for unit in shards.values()),
         'Slurm task must be an admitted node in its numbered parent')
    return plan['selected'][node]
