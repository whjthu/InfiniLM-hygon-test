"""Immutable request, placement and shape for the two-parent component backfill."""
from collections import defaultdict
from hashlib import sha256
import re


REQUEST = "cde5c3eabf0943a8b303a3f260963588"
LABEL = "fleet-qwen-projection-weight-layout-v1"
PARENTS = ("1087656", "1103606")
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
KNOWN_FOREIGN_KFD = frozenset(("f11r2n15", "f12r2n12"))


def owners(ledger):
    result = {}
    for group, row in ledger["groups"].items():
        allocations = row["allocations"]
        if set(allocations) != set(PARENTS):
            raise ValueError("Original ledger parents differ")
        for parent, nodes in allocations.items():
            for node in nodes:
                if node in result:
                    raise ValueError("Duplicate node in original ledger")
                result[node] = (parent, group)
    if len(result) != 80 or any(sum(item[0] == parent for item in result.values()) != 40 for parent in PARENTS):
        raise ValueError("Expected disjoint 40-node allocations")
    return result


def shards(selected, roster):
    if (not selected or set(selected) & KNOWN_FOREIGN_KFD or set(selected)-roster.keys()
            or any(len(pci) != 8 or len(set(pci)) != 8 or any(not PCI.fullmatch(item) for item in pci)
                   for pci in selected.values())):
        raise ValueError("Selected card or node membership differs")
    by_group = defaultdict(list)
    for node in sorted(selected):
        by_group[roster[node]].append(node)
    result = {}
    for (parent, group), nodes in sorted(by_group.items()):
        for offset in range(0, len(nodes), 4):
            label = f"qwenweight-{REQUEST}-{parent}-{group}-{offset//4:02d}"
            req = sha256(label.encode("ascii")).hexdigest()[:32]
            result[label] = dict(parent=parent, nodes=nodes[offset:offset+4], request=req)
    return result
