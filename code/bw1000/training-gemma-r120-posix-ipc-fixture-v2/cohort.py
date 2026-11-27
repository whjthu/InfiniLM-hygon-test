"""Bind two independent native caches and retire only the registered directories."""

import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provision import ProvisionedPool, need, verify_provider


class Leases:
    order = ("multiprocess", "multiprocessing")

    def __init__(self, root, request, *, posix=True):
        self.items = {name: ProvisionedPool(root, request, posix=posix) for name in self.order}

    def attach(self, pins):
        need(set(pins) == set(self.order), "EXACT_DUAL_PROVIDER_BINDINGS")
        providers = {}
        for name in self.order:
            native = tuple(importlib.import_module(f"{name}{suffix}")
                           for suffix in ("", ".util", ".process", ".connection"))
            provider = native[0]
            config = provider.current_process()._config
            need(type(config) is dict and config.get("tempdir") is None
                 and not provider.active_children(), "NEW_IDLE_PROVIDER_CACHE")
            providers[name] = (native, config)
        first, second = (providers[n][1] for n in self.order)
        need(first is not second, "INDEPENDENT_PROVIDER_CACHES")
        # Confirm both source contracts before the first directory is created.
        verified = {name: verify_provider(*native, pins[name])
                    for name, (native, _) in providers.items()}
        created = []
        for name in self.order:
            lease = self.items[name]
            lease.create(peers=tuple(created))
            created.append(lease)
            lease.attach(*providers[name][0], pins[name])
        return verified

    def creation(self):
        return {name: lease.evidence() for name, lease in self.items.items()}

    def retire(self, *, children_reaped):
        remaining = list(self.items.values())
        for lease in remaining:
            peers = tuple(peer for peer in remaining if peer is not lease)
            lease.preflight(children_reaped=children_reaped, peers=peers)
        for index, lease in enumerate(remaining):
            lease.finish(children_reaped=children_reaped, peers=tuple(remaining[index + 1:]))

    def results(self):
        return {name: lease.evidence() if hasattr(lease, "identity")
                else {"state": "UNCREATED", "directory_removals": 0}
                for name, lease in self.items.items()}

    def close(self):
        failure = None
        for lease in self.items.values():
            try:
                lease.close()
            except Exception as exc:
                failure = exc
        if failure is not None:
            raise failure


def manager(lease, observer, records, name, stamp):
    result = records[name] = {"queue_roundtrip": False}
    with lease.provider.get_context().Manager() as owner:
        endpoint = Path(owner.address)
        process = owner._process
        result.update(pid=process.pid, startticks=stamp(process.pid))
        lease.verify()
        identity = endpoint.lstat()
        need(endpoint.parent.resolve(strict=True) == lease.child
             and stat.S_ISSOCK(identity.st_mode) and identity.st_uid == os.getuid()
             and identity.st_dev == lease.identity[0] and process.is_alive(),
             "EXACT_REGISTERED_MANAGER_ENDPOINT")
        result["registered_endpoint"] = True
        observer.sample("MANAGER_LIVE")
        queue = owner.Queue()
        queue.put(42)
        need(queue.get(timeout=5) == 42, "ACTUAL_MANAGER_QUEUE")
        del queue
        result["queue_roundtrip"] = True
    result["exitcode"] = process.exitcode
    result["provider_children_after"] = len(lease.provider.active_children())
    need(result["exitcode"] == 0 and not process.is_alive()
         and result["provider_children_after"] == 0, "ACTUAL_MANAGER_AND_CHILDREN_REAPED")


def check(folder, request, host_sha, lifecycle_sha, cleanup_sha):
    from readback import raw, review

    native = review(folder, request, lifecycle_sha, cleanup_sha)
    host_raw = raw(folder / "host.json", 16384)
    host = json.loads(host_raw)
    guest_raw = raw(folder / "guest.json", 16384)
    guest = json.loads(guest_raw)
    creation_raw = raw(folder / "registered-ipc-creation.json", 4096)
    creation = json.loads(creation_raw)
    need(sha256(host_raw).hexdigest() == host_sha and sha256(guest_raw).hexdigest() == host["guest_SHA256"]
         and host["request"] == guest["request"] == creation["request"] == request
         and host["step"] == guest["step"], "BOUND_HOST_GUEST_CREATION_AND_CLEANUP")
    keys = set(Leases.order)
    provider_pins = json.loads((Path(__file__).resolve().parent / "PROVIDER_BINDING.json").read_bytes())
    bindings = guest["deployed_provider_sources"]
    need(set(provider_pins) == set(bindings) == keys, "EXACT_PROVIDER_SOURCE_DENOMINATOR")
    for name in keys:
        parts = provider_pins[name]
        got = bindings[name]
        need(set(got) == set(parts) | {"semantics"}
             and got["semantics"] == dict(getter_cache_short_circuit=True, child_config_copy=True,
                                           unix_listener_uses_cache=True)
             and all(got[part] == {key: value for key, value in pin.items() if key != "path"}
                     for part, pin in parts.items()), "EXACT_OBSERVED_PROVIDER_SOURCE_PINS")
    before, after = creation["creation"], guest["registered_cleanup"]
    need(guest["registered_creation"] == before and set(before) == set(after) == set(guest["managers"]) == keys
         and len({(row["device"], row["inode"]) for row in before.values()}) == 2
         and len({(row["pid"], row["startticks"]) for row in guest["managers"].values()}) == 2,
         "EXACT_TWO_DIRECTORY_AND_MANAGER_IDENTITIES")
    fields = ("creator_pid", "creator_startticks", "inode", "uid", "device", "mount_id", "name_SHA256")
    for name in keys:
        manager_row = guest["managers"][name]
        need(before[name]["recorded_creation"] is before[name]["provider_cache_bound"] is True
             and before[name]["directory_removals"] == 0
             and all(before[name][key] == after[name][key] for key in fields)
             and after[name]["directory_removals"] == 1 and after[name]["state"] == "RETIRED"
             and manager_row["queue_roundtrip"] is manager_row["registered_endpoint"] is True
             and manager_row["exitcode"] == manager_row["provider_children_after"] == 0,
             "INVALID_REGISTERED_RETIREMENT_OR_MANAGER_OUTCOME")
    need(native["passed"] is host["passed"] is guest["passed"] is True
         and guest["planned_dataset_map_rows"] == guest["actual_dataset_map_rows"] == 2
         and guest["GPU"] == host["GPU"] == guest["actual_training_rows"] == guest["weight_payloads"] == 0
         and host["host_managers_absent"] == dict.fromkeys(keys, True) and host["host_marker_only_after"],
         "EXACT_ZERO_GRES_IPC_FIXTURE_RESULT")
    return native | {"step": host["step"], "planned_rows": 2, "completed_rows": 2, "registered_rmdirs": 2,
                     "mapped_provider_source_files": 6, "guest_SHA256": host["guest_SHA256"], "host_SHA256": host_sha}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    for name in ("request", "host-sha256", "lifecycle-sha256", "cleanup-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.request, args.host_sha256, args.lifecycle_sha256,
                           args.cleanup_sha256), sort_keys=True))
