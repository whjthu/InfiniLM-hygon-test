"""Two represented O01 parents and exact frozen representative node pool."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re

from contract import FIRST_ACTIVE, WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS, assign_cohorts, grid


LABEL = 'O01-BF16-dependent-replay-depth9-15-crossparent-v1'
REQUEST = 'b37e8f186c074969921d3118769a53f0'
RUN = LABEL + '-' + REQUEST
PARENTS = ('1087656','1103606')
ZONES = ('O01',*(f'I{i:02d}' for i in range(1,4)),*(f'T{i:02d}' for i in range(1,13)))
LEDGER_SHA = '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'
PILOT_SHA = '8cb8bcb68ca7c4b308d94358c1b0987597f06accdbf7cf7e346c3b41600ea5c8'
FLEET_SHA = '8fce0f54420b4b9dbf9bef720f18e4066f02fc548bf596a58f0822f9ae646d08'
EXCLUDED_NODES = frozenset(('f11r2n15','f11r4n09','f11r4n10','f11r4n19','f12r2n12'))
REPRESENTATIVE_NODES = {"f12r3n17":"1087656","f12r3n18":"1087656","f12r2n10":"1103606","f12r2n13":"1103606"}
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
    need(type(selected) is dict and set(selected)==set(REPRESENTATIVE_NODES)
         and not EXCLUDED_NODES.intersection(selected),'selected nodes differ from O01 representative pool')
    by_parent={parent:[] for parent in PARENTS}
    for node,item in sorted(selected.items()):
        parent=REPRESENTATIVE_NODES[node]
        pci,before=item['PCI'],item['before_CPU_step']
        need((item['parent'],item['zone'])==owners[node]==(parent,'O01')
             and type(pci) is list and len(pci)==len(set(pci))==8
             and all(isinstance(device,str) and PCI.fullmatch(device) for device in pci)
             and isinstance(before,str) and re.fullmatch(parent+r'\.[0-9]+',before),
             'original O01 ledger parent, PCI or CPU prerequisite differs')
        by_parent[parent].append(node)
    need(all(len(members)==2 for members in by_parent.values()),'both O01 parent cohorts require eligible nodes')
    output={}
    for parent,nodes in by_parent.items():
        label=f'O01-bf16-dependent-depth-{REQUEST[:12]}-{parent}'
        output[label]=dict(parent=parent,zone='O01',nodes=nodes,
                           request=sha256((REQUEST+':'+label).encode()).hexdigest()[:32])
    return output


def validate_plan(plan,owners):
    epoch=plan['campaign_epoch']
    grid(plan,0)
    need(plan['request']==REQUEST and type(epoch) is float and math.isfinite(epoch)
         and plan['shared_start_epoch']==epoch+FIRST_ACTIVE[0] and plan['deadline_epoch']==epoch+WORKER_SECONDS
         and plan['pilot_SHA256']==PILOT_SHA and plan['fleet_terminal_SHA256']==FLEET_SHA
         and all(HEX.fullmatch(plan[key]) for key in ('source_SHA256','before_snapshot_SHA256'))
         and re.fullmatch(r'[a-f0-9]{40}',plan['publication']),'original campaign or fixed absolute epoch differs')
    shards=placement(plan['selected'],owners)
    need(shards==plan['shards'] and assign_cohorts(shards)==plan['cohort_by_shard'],
         'derived fixed O01 parent cohort membership differs')
    return shards


def validate_worker_plan(plan,node,parent,step):
    shards=validate_plan(plan,staged_ledger())
    need(step.isdecimal() and any(unit['parent']==parent and node in unit['nodes'] for unit in shards.values()),
         'Slurm task must be an admitted node in its numbered parent')
    return plan['selected'][node]
