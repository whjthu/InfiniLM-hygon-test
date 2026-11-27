"""Rank startup custody and supervisor-only retirement for the native Swift run."""

from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from provision import ProvisionedPool, mounting, need, physical, process_stamp, verify_provider

PROVIDERS = ("multiprocess", "multiprocessing")


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def save(path, value):
    raw = encode(value)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def read(path, digest=None, cap=16384):
    need(path.is_file() and not path.is_symlink(), "R120_IPC_EXPECTED_REGULAR_FILE")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    need(0 < len(raw) <= cap and (digest is None or sha256(raw).hexdigest() == digest), "R120_IPC_FILE_PIN")
    return raw


class Owner:
    def __init__(self, short, root, request):
        self.short, self.root, self.request = short, root, request
        self.leases = []
        self.assignment = {}

    def reserve(self):
        for rank in range(8):
            records = {}
            self.assignment[str(rank)] = records
            for provider in PROVIDERS:
                lease = ProvisionedPool(self.short, self.request)
                self.leases.append(lease)
                lease.create(peers=tuple(self.leases[:-1]))
                records[provider] = {"name": lease.child.name, "physical": list(lease.identity), "mount": lease.mount}
        return save(self.root / "ipc-rank-assignments.json",
                    {"request": self.request, "parent": os.getpid(), "parent_startticks": process_stamp(),
                     "assignments": self.assignment})

    def verified_ranks(self, rank_owners):
        need(set(rank_owners) == set(range(8)), "R120_IPC_EIGHT_RANK_OWNERS")
        result = {}
        for rank, owner in rank_owners.items():
            handoff = json.loads(read(self.root / f"ipc-rank-{rank}.json"))
            need(handoff["request"] == self.request and handoff["rank"] == rank
                 and (handoff["pid"], handoff["startticks"]) == (owner["PID"], owner["startticks"])
                 and handoff["registered"] == self.assignment[str(rank)] and handoff["status"] == "READY",
                 "R120_IPC_RANK_START_AND_CALLBACK_DISAGREE")
            result[str(rank)] = handoff
        return result

    def retire(self, *, reaped, report):
        report.update(planned=16, registered=sum(hasattr(x, "identity") for x in self.leases), removed=0)
        need(reaped is True and not (Path("/proc/self/task") / str(os.getpid()) / "children").read_text().split(),
             "R120_IPC_NO_LIVE_DIRECT_OR_REGISTERED_CHILDREN")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        if not self.leases:
            need({p.name for p in self.short.iterdir()} == {".scope"}, "R120_IPC_UNREGISTERED_RESIDUE_RETAINED")
            return
        need(report["registered"] == len(self.leases), "R120_IPC_INCOMPLETE_CREATION_RETAINED")
        pending = list(self.leases)

        def check(lease):
            lease.verify()
            need(lease.creator_pid == os.getpid() and lease.creator_startticks == process_stamp()
                 and physical(os.fstat(lease.root_fd)) == lease.parent
                 and physical(os.fstat(lease.child_fd)) == lease.identity
                 and mounting(lease.root_fd) == mounting(lease.child_fd) == lease.mount
                 and not os.listdir(lease.child_fd), "R120_IPC_NONEMPTY_OR_PROVENANCE_CHANGED")
            fd = os.open(lease.child.name, flags, dir_fd=lease.root_fd)
            try:
                need(physical(os.fstat(fd)) == lease.identity and mounting(fd) == lease.mount,
                     "R120_IPC_FINAL_NAME_IDENTITY_CHANGED")
            finally:
                os.close(fd)

        def exact_pending():
            need(set(os.listdir(self.short)) == {".scope", *(item.child.name for item in pending)},
                 "R120_IPC_UNREGISTERED_RESIDUE_RETAINED")

        exact_pending()
        for lease in pending:
            check(lease)
        for lease in self.leases:
            exact_pending()
            check(lease)
            os.rmdir(lease.child.name, dir_fd=lease.root_fd)
            pending.remove(lease)
            report["removed"] += 1
        exact_pending()

    def close(self):
        for lease in self.leases:
            lease.close()


def rank_on_entry():
    """Run only for the exact torchrun SFT script, never fork/spawn helper entrypoints."""
    root_value = os.environ.get("GEMMA_IPC_OPERATION")
    if not root_value or not sys.argv or sys.argv[0] != os.environ.get("GEMMA_IPC_RANK_ENTRY"):
        return
    rank = os.environ.get("RANK", "")
    root = Path(root_value)
    valid = re.fullmatch("[0-7]", rank)
    try:
        need(valid and os.environ["WORLD_SIZE"] == "8" and rank == os.environ["LOCAL_RANK"],
             "R120_IPC_EXACT_TORCHRUN_RANK")
        stage = Path(__file__).resolve().parent
        approval = json.loads(read(stage / "CLAIM.json"))
        entry = approval["native_rank_entry"]
        need(entry["path"] == sys.argv[0]
             and len(read(Path(entry["path"]), entry["SHA256"], 65536)) == entry["bytes"],
             "R120_IPC_EXACT_NATIVE_SWIFT_ENTRY")
        raw = read(root / "ipc-rank-assignments.json", os.environ["GEMMA_IPC_ASSIGNMENTS_SHA256"])
        witness = json.loads(raw)
        request = approval["training_request_id"]
        short = Path(approval["private_tmp_alias"])
        need(witness["request"] == request and root.name == approval["label"]
             and short.resolve(strict=True) == short and
             all(os.environ.get(key) == str(short) for key in ("TMPDIR", "TEMP", "TMP")),
             "R120_IPC_EXACT_RANK_MOUNT_AND_SCOPE")
        marker = short / ".scope"
        ms = marker.lstat()
        need(stat.S_ISREG(ms.st_mode) and ms.st_nlink == 1 and physical(ms)[2:] == (os.getuid(), 0o600)
             and read(marker, cap=64) == (request + "\n").encode(), "R120_IPC_EXACT_RANK_MARKER")
        pinned = json.loads(read(stage / "PROVIDER_BINDING.json"))
        registered = witness["assignments"][rank]
        for key in PROVIDERS:
            module, util, proc, conn = (importlib.import_module(key + suffix)
                                        for suffix in ("", ".util", ".process", ".connection"))
            verify_provider(module, util, proc, conn, pinned[key])
            record = registered[key]
            need(re.fullmatch("pymp-[A-Za-z0-9_-]{8}", record["name"]) is not None, "R120_IPC_EXACT_CREATED_NAME")
            child = short / record["name"]
            fd = os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                need(list(physical(os.fstat(fd))) == record["physical"] and mounting(fd) == record["mount"],
                     "R120_IPC_RANK_DIRECTORY_IDENTITY")
            finally:
                os.close(fd)
            cache = module.current_process()._config
            need(type(cache) is dict and cache.get("tempdir") is None, "R120_IPC_EARLY_PARENT_CACHE_REQUIRED")
            cache["tempdir"] = str(child)
            need(util.get_temp_dir() == str(child), "R120_IPC_WRONG_ACTIVE_PROVIDER_CACHE")
        save(root / f"ipc-rank-{rank}.json", {"request": request, "rank": int(rank), "pid": os.getpid(),
             "startticks": process_stamp(), "status": "READY", "registered": registered})
    except BaseException as exc:
        try:
            reason = str(exc) if type(exc) is ValueError and re.fullmatch("[A-Z0-9_]{1,85}", str(exc)) else None
            if valid:
                save(root / f"ipc-rank-{rank}-startup-failure.json", {"class": type(exc).__name__, "reason": reason})
        finally:
            # sitecustomize normally warns and continues. A matched rank must stop.
            os._exit(74)


def callback_gate(rank, args):
    need(getattr(args, "use_logits_to_keep", None) is False, "R120_IPC_EFFECTIVE_LOGITS_TO_KEEP_MUST_BE_FALSE")
    root = Path(os.environ["GEMMA_IPC_OPERATION"])
    witness = json.loads(read(root / f"ipc-rank-{rank}.json"))
    assignment = json.loads(read(root / "ipc-rank-assignments.json", os.environ["GEMMA_IPC_ASSIGNMENTS_SHA256"]))
    need(witness["request"] == assignment["request"] and witness["registered"] == assignment["assignments"][str(rank)]
         and witness["rank"] == rank and witness["pid"] == os.getpid()
         and witness["startticks"] == process_stamp() and witness["status"] == "READY",
         "R120_IPC_MISSING_SAME_PROCESS_STARTUP")
    for key in PROVIDERS:
        cache = importlib.import_module(key).current_process()._config
        need(cache.get("tempdir") == str(Path(os.environ["TMPDIR"]) / witness["registered"][key]["name"]),
             "R120_IPC_CACHED_DIRECTORY_CHANGED_BEFORE_OPTIMIZER")
