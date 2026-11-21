"""The four-node O01 replication contract; no collective or borrowed group."""

from hashlib import sha256
import math
import re


LABEL = 'O01-resident-BF16-graph64-32card-v1'
REQUEST = '1f31f75efa1c9568e8f6b3173a383d8f'
RUN = LABEL + '-' + REQUEST
PARENTS = ('1087656','1103606')
O01 = {
    '1087656': {'f12r3n17','f12r3n18','f12r3n19','f12r3n20'},
    '1103606': {'f12r2n10','f12r2n11','f12r2n12','f12r2n13'},
}
PILOT_NODE = 'f12r2n13'
PREFERRED = {'1087656': ('f12r3n17','f12r3n18'), '1103606': ('f12r2n10','f12r2n11')}
LEDGER_SHA = '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'
WORKER_SECONDS, SLURM_SECONDS, CONTROLLER_SECONDS = 125, 140, 180
PCI = re.compile(r'[a-f0-9]{4}:[a-f0-9]{2}:[01][a-f0-9]\.[0-7]')
HEX = re.compile(r'[a-f0-9]{64}')


def need(condition, why):
    if not condition:
        raise ValueError(why)


def shard_label(parent):
    need(parent in PARENTS, 'parent is outside the original O01 allocations')
    return 'graph64-32-' + REQUEST[:12] + '-' + parent


def placement(selected):
    need(type(selected) is dict and len(selected) == 4 and PILOT_NODE not in selected,
         'accept exactly four independent nodes; exclude the old singleton pilot')
    result = {parent:[] for parent in PARENTS}
    for node, item in selected.items():
        parent = item.get('parent')
        pci = item.get('PCI')
        need(parent in O01 and node in O01[parent] and item.get('zone') == 'O01'
             and type(pci) is list and len(pci) == len(set(pci)) == 8
             and all(isinstance(device,str) and PCI.fullmatch(device) for device in pci)
             and isinstance(item.get('before_CPU_step'),str) and
             re.fullmatch(parent + r'\.[0-9]+', item['before_CPU_step']),
             'original CPU-only step, PCI or O01 membership differs')
        result[parent].append(node)
    need(all(len(nodes) == 2 for nodes in result.values()), 'two nodes per approved parent required')
    return {p:sorted(nodes) for p,nodes in result.items()}


def frozen_ledger(raw, decoded):
    need(sha256(raw).hexdigest() == LEDGER_SHA
         and {parent:set(nodes) for parent,nodes in decoded['groups']['O01']['allocations'].items()} == O01,
         'O01 actual ledger pin differs')
    return decoded


def validate_plan(plan):
    need(plan['request'] == REQUEST and type(plan['campaign_epoch']) is float
         and math.isfinite(plan['campaign_epoch']) and plan['shared_start_epoch'] == plan['campaign_epoch']+60
         and plan['deadline_epoch'] == plan['campaign_epoch']+WORKER_SECONDS
         and all(HEX.fullmatch(plan[key]) for key in ('source_SHA256','before_snapshot_SHA256','pilot_SHA256'))
         and re.fullmatch(r'[a-f0-9]{40}',plan['publication']), 'replication request or shared absolute epoch differs')
    return placement(plan['selected'])


def validate_worker_plan(plan, node, parent, step):
    groups = validate_plan(plan)
    need(parent in groups and node in groups[parent] and step.isdecimal(),
         'Slurm task is not an admitted node of its parent shard')
    return plan['selected'][node]
