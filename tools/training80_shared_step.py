"""Admit an engineering workload inside its own single-node, non-overlapping DCU step."""

import argparse
import ctypes
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import time

from read_training_slurm_membership import allocated_dcu, fields, query


PCI = re.compile(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')
DRM = Path('/sys/class/drm')
DEV = Path('/dev/dri')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_plan(plan):
    need(re.fullmatch(r'[0-9]+', plan['parent']) and re.fullmatch(r'[a-z0-9]+', plan['node'])
         and re.fullmatch(r'[0-9a-f]{32}', plan['request_id']), 'invalid owned step identity')
    need(plan['workload_scope'] == 'engineering' and type(plan['cards']) is int and 1 <= plan['cards'] < 8,
         'shared nodes require an engineering workload smaller than eight DCUs')
    need(type(plan['deadline_epoch']) in (int, float) and math.isfinite(plan['deadline_epoch'])
         and type(plan['duration_limit_s']) is int and plan['duration_limit_s'] > 0,
         'shared step requires the original finite deadline')
    inventory = plan['node_PCIs']
    need(type(inventory) is list and len(inventory) == len(set(inventory)) == 8
         and all(PCI.fullmatch(pci) for pci in inventory), 'expected eight pinned PCI identities')
    peers = plan['peer_step_PCIs']
    need(isinstance(peers, dict) and peers and all(type(pcis) is list for pcis in peers.values())
         and all(re.fullmatch(re.escape(plan['parent']) + r'\.[0-9]+', step) for step in peers),
         'explicit placements for all colocated steps are required')
    occupied = [pci for pcis in peers.values() for pci in pcis]
    need(len(occupied) == len(set(occupied)) and set(occupied) <= set(inventory)
         and plan['cards'] <= len(inventory) - len(occupied), 'overlapping or insufficient pinned PCI capacity')
    return set(occupied)


def srun_args(plan, work, *, cpus, memory_mib, minutes):
    """Compose the scheduling request. The workload calls guard() in this same step."""
    validate_plan(plan)
    need(all(type(value) is int and value > 0 for value in (cpus, memory_mib, minutes)),
         'exact CPU, memory and time allocations must be positive')
    need(minutes * 60 <= plan['duration_limit_s'], 'Slurm step exceeds the original budget')
    need(isinstance(work, (tuple, list)) and work and PurePosixPath(work[0]).is_absolute(),
         'frozen workload must use an absolute executable')
    return ['srun', '--jobid=' + plan['parent'], '--job-name=training-share-' + plan['request_id'],
            '--exact', '--immediate=5', '--nodes=1', '--ntasks=1', '--nodelist=' + plan['node'],
            f'--cpus-per-task={cpus}', f'--mem={memory_mib}M', f'--time={minutes}',
            f'--gres=dcu:{plan["cards"]}', *work]


def srun_env(environ):
    """Keep the caller's runtime environment without inherited step-sharing options."""
    excluded = {'SLURM_OVERLAP', 'SLURM_OVERCOMMIT', 'SLURM_OVERSUBSCRIBE', 'SRUN_OVERLAP',
                'SLURM_STEP_GRES', 'SLURM_MEM_PER_CPU', 'SLURM_MEM_PER_GPU', 'SLURM_MEM_PER_NODE',
                'SLURM_GPUS', 'SLURM_GPUS_PER_NODE', 'SLURM_GPUS_PER_SOCKET', 'SLURM_GPUS_PER_TASK',
                'SLURM_TRES_PER_TASK', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                'GPU_DEVICE_ORDINAL'}
    return {key: value for key, value in environ.items() if key not in excluded}


def step_inventory(plan, run_query):
    """Return live local step identities and counts without interpreting numeric GPU ordinals."""
    local = {}
    parent, node = plan['parent'], plan['node']
    rows = run_query('scontrol', 'show', 'step', parent, '--oneliner')
    need(rows, 'live Slurm step inventory is absent')
    for line in rows:
        row = fields(line)
        step = row.get('StepId', '')
        if step in (parent + '.batch', parent + '.extern'):
            continue
        need(re.fullmatch(re.escape(parent) + r'\.[0-9]+', step), 'unknown Slurm step')
        hosts = run_query('scontrol', 'show', 'hostnames', row['NodeList'])
        need(len(hosts) == len(set(hosts)) == int(row['Nodes']), 'ambiguous Slurm step location')
        if node not in hosts:
            continue
        # Aggregate DCU counts for a multi-node peer cannot prove its local allocation.
        need(len(hosts) == 1 and step not in local and row['State'] == 'RUNNING',
             'colocated Slurm step has no exact stable single-node allocation')
        resources = row.get('TRES')
        if resources in (None, '', 'N/A', '(null)'):
            resources = row.get('AllocTRES')
        count = allocated_dcu(resources)
        need(count is not None and 0 <= count <= 8, 'unknown colocated DCU allocation')
        local[step] = count, row.get('Name')
    return local


def hip_pcis():
    hip = ctypes.CDLL('libamdhip64.so')
    count = ctypes.c_int()
    get_count = hip.hipGetDeviceCount
    get_count.argtypes, get_count.restype = (ctypes.POINTER(ctypes.c_int),), ctypes.c_int
    need(get_count(ctypes.byref(count)) == 0 and 0 < count.value <= 8, 'HIP device inventory is unavailable')
    get_pci = hip.hipDeviceGetPCIBusId
    get_pci.argtypes, get_pci.restype = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int), ctypes.c_int
    found = []
    for index in range(count.value):
        buffer = ctypes.create_string_buffer(32)
        need(get_pci(buffer, len(buffer), index) == 0, 'HIP device has no PCI address')
        address = buffer.value.decode('ascii').lower()
        found.append('0000:' + address if address.count(':') == 1 else address)
    need(all(PCI.fullmatch(pci) for pci in found), 'invalid HIP PCI address')
    return found


def drm_access(*, drm=DRM, dev=DEV):
    """Check effective DRM access from within the new step's own cgroup."""
    inventory, render, primary = [], set(), set()
    for entry in drm.iterdir():
        if not re.fullmatch(r'(renderD|card)[0-9]+', entry.name):
            continue
        pci = (entry / 'device').resolve(strict=True).name.lower()
        node = dev / entry.name
        # A non-PCI primary node is irrelevant, but may never establish a render identity.
        is_render = entry.name.startswith('renderD')
        if not PCI.fullmatch(pci) and not is_render:
            continue
        need(PCI.fullmatch(pci), 'DRM render node has no PCI identity')
        if is_render:
            inventory.append(pci)
        try:
            need(stat.S_ISCHR(node.lstat().st_mode), 'DRM device path is not a character device')
            fd = os.open(node, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as error:
            if error.errno in (errno.EPERM, errno.EACCES, errno.ENOENT):
                continue
            raise
        else:
            os.close(fd)
            (render if is_render else primary).add(pci)
    return inventory, render, primary


def guard(plan, *, environ=None, observer_uid=None, hostname=socket.gethostname, run_query=query,
          visible_devices=hip_pcis, drm_devices=drm_access, clock=time.time):
    """Authorize only the actual new step's assigned, disjoint physical PCI set."""
    occupied = validate_plan(plan)
    env = os.environ if environ is None else environ
    step_index = env.get('SLURM_STEP_ID', '')
    if env.get('SLURM_JOB_ID') != plan['parent'] or not step_index.isdecimal():
        raise ValueError('workload is outside the owned numbered Slurm step')
    own_step = plan['parent'] + '.' + step_index
    need(hostname().split('.', 1)[0] == plan['node'] and own_step not in plan['peer_step_PCIs'],
         'workload node or step identity differs')
    deadline = plan['deadline_epoch']
    need(0 < deadline - clock() <= plan['duration_limit_s'],
         'shared admission is past or outside its original budget')
    uid = os.getuid() if observer_uid is None else observer_uid
    rows = run_query('scontrol', 'show', 'job', plan['parent'], '--oneliner')
    need(len(rows) == 1, 'owned parent allocation is unavailable')
    parent = fields(rows[0])
    need(parent.get('JobId') == plan['parent'] and parent.get('JobState') == 'RUNNING'
         and re.fullmatch(r'[^()]+\(' + str(uid) + r'\)', parent.get('UserId', '')),
         'Slurm parent is not running under the current owner')
    initial = step_inventory(plan, run_query)
    own = initial.get(own_step)
    need(own == (plan['cards'], 'training-share-' + plan['request_id'])
         and {step: count for step, (count, _) in initial.items() if step != own_step}
         == {step: len(pcis) for step, pcis in plan['peer_step_PCIs'].items()},
         'all colocated step counts and the original owner must match their pinned leases')
    visible = visible_devices()
    inventory, render, primary = drm_devices()
    assigned = set(visible)
    need(sorted(inventory) == sorted(plan['node_PCIs']) and len(visible) == len(assigned) == plan['cards']
         and assigned <= set(inventory) and assigned.isdisjoint(occupied)
         and render == assigned and primary <= assigned, 'actual HIP/DRM device access is not isolated')
    need(initial == step_inventory(plan, run_query) and clock() < deadline, 'colocated leases changed during admission')
    return dict(request_id=plan['request_id'], admission_step=own_step, node=plan['node'],
                assigned_PCIs=visible, checked_peer_steps=sorted(plan['peer_step_PCIs']),
                workload_scope=plan['workload_scope'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('plan_sha256')
    args = parser.parse_args()
    with args.plan.open('rb') as source:
        raw = source.read(16385)
    need(len(raw) <= 16384 and sha256(raw).hexdigest() == args.plan_sha256, 'published shared-step plan differs')
    print(json.dumps(guard(json.loads(raw)), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
