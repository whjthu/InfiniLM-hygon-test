"""Bind the one runtime-assigned PCI to its accessible KFD topology entry."""
import hashlib
import os
from pathlib import Path
import re


PCI = re.compile(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')


def pci_for_render(render):
    return Path(os.readlink(Path('/sys/class/drm') / render / 'device')).name.lower()


def assigned_kfd(pci, *, nodes=Path('/sys/class/kfd/kfd/topology/nodes'), pci_lookup=pci_for_render):
    if not isinstance(pci, str) or not PCI.fullmatch(pci):
        raise ValueError('invalid assigned PCI')
    entries = list(nodes.iterdir())
    if not 0 < len(entries) <= 16:
        raise ValueError('unexpected per-node KFD topology size')
    matches, denied = [], []
    for node in entries:
        try:
            text = (node / 'properties').read_text()
        except PermissionError:
            # Unallocated GPUs may deliberately hide their topology metadata.
            # They are never accepted as the assigned device or read for gpu_id.
            denied.append(node.name)
            continue
        properties = dict(row.split(maxsplit=1) for row in text.splitlines())
        if 'drm_render_minor' not in properties:
            continue
        minor = int(properties['drm_render_minor'])
        if minor < 128:  # CPU topology entries have no DRM render device.
            continue
        render = 'renderD' + str(minor)
        if pci_lookup(render) != pci:
            continue
        domain, bus, device = pci.split(':')
        slot, function = device.split('.')
        location = int(bus, 16) << 8 | int(slot, 16) << 3 | int(function, 16)
        if int(properties['domain']) != int(domain, 16) or int(properties['location_id']) != location:
            raise ValueError('assigned DRM/KFD PCI mapping differs')
        gpu_id = int((node / 'gpu_id').read_text())
        if gpu_id <= 0:
            raise ValueError('assigned KFD GPU identity is absent')
        matches.append(dict(pci=pci, gpu_id=gpu_id, topology_node=node.name,
                            properties_SHA256=hashlib.sha256(text.encode()).hexdigest()))
    if len(matches) != 1:
        raise ValueError('expected exactly one accessible assigned KFD device')
    return dict(**matches[0], denied_unassigned_properties=denied,
                other_devices_gpu_id_read=False,
                qualification='Only the runtime-assigned device is bound; no claim of access to all eight KFD IDs.')
