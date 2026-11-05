"""Materialize only pinned GLM prefix sources and validate the effective compiler dependencies."""

from hashlib import sha256
import posixpath
from pathlib import Path, PurePosixPath
import shlex

from bind_prefix_native_rebuild_argv import DEST, MIRROR, mirror
from derive_accepted_prefix_bridge import derive as derive_bridge
from derive_accepted_prefix_headers import HEADERS, add_access

SUPPORT = ("accepted_prefix_native.hpp", "accepted_prefix_ticket.hpp", "accepted_prefix_lengths.hpp")


ORIGINAL_ALIASES = {
    '/refactor/candidates/glm-M16-r8861/tree/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/engine/distributed/../../parallel/topology.hpp':
        '/refactor/candidates/glm-M16-r8861/tree/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/parallel/topology.hpp',
    '/refactor/candidates/glm-M16-r8861/tree/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/layers/rotary_embedding/../../weights/weights.hpp':
        '/refactor/candidates/glm-M16-r8861/tree/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/weights/weights.hpp',
    '/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/engine/distributed/../../parallel/topology.hpp':
        '/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/parallel/topology.hpp',
    '/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/layers/rotary_embedding/../../weights/weights.hpp':
        '/refactor/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/weights/weights.hpp',
}

MIRRORED_ALIASES = {
    str(MIRROR) + alias.removeprefix("/refactor"): mirror(target)
    for alias, target in ORIGINAL_ALIASES.items()
}


def original_mirror(path):
    return mirror(ORIGINAL_ALIASES.get(path, path))


def digest(raw):
    return sha256(raw).hexdigest()


def dependencies(raw, target, aliases=None):
    line = raw.decode("utf-8").replace("\\\n", " ")
    lhs, sep, rhs = line.partition(": ")
    if not sep or shlex.split(lhs) != [target]:
        raise ValueError("compiler dependency target differs")
    result = set(shlex.split(rhs))
    if not result or any(not PurePosixPath(name).is_absolute() or (".." in PurePosixPath(name).parts and name not in (aliases or {}))
                         for name in result):
        raise ValueError("compiler dependency path differs")
    return result


def preflight(plan, dep_receipt, reader):
    if (plan["destination"] != str(DEST) or plan["build_authorized"] is not False
            or dep_receipt["failures"] or len(plan["compiles"]) != 8):
        raise ValueError("incomplete prefix rebuild input")
    pins = dep_receipt["dependency_sha256"]
    expected, original_paths, rewritten = {}, set(), {}
    for row in plan["compiles"]:
        original_target = row["original_object"]
        raw = reader(row["original_MD"])
        if digest(raw) != row["original_MD_sha256"]:
            raise ValueError("original dependency list changed")
        paths = dependencies(raw, original_target, ORIGINAL_ALIASES)
        if row["original_source"] not in paths or pins.get(row["original_source"]) != row["source_sha256"]:
            raise ValueError("original compile source pin differs")
        headers = {h: mirror(h) for h in paths if PurePosixPath(h).name in HEADERS}
        if headers != row["expected_rewritten_headers"] or not headers:
            raise ValueError("original changed-header inventory differs")
        for name, path in headers.items():
            if path == name or pins.get(name) != HEADERS[PurePosixPath(name).name]:
                raise ValueError("original class header pin differs")
            rewritten[name] = path
        current = {original_mirror(name) for name in paths}
        source_dir = PurePosixPath(mirror(row["original_source"])).parent
        if row["source_derivation"] == "prefix_bridge":
            current.update(str(source_dir / name) for name in SUPPORT)
        output = row["compile_argv"][-1]
        if output in expected or not PurePosixPath(output).is_relative_to(DEST / "obj"):
            raise ValueError("candidate dependency target collision")
        expected[output] = sorted(current)
        original_paths.update(paths)
    if original_paths != pins.keys() or {PurePosixPath(p).name for p in rewritten} != HEADERS.keys():
        raise ValueError("missing or extra pinned original dependencies")
    if len({path for path in rewritten.values()}) != len(rewritten):
        raise ValueError("rewritten class paths collide")
    return expected


def build_mirror(plan, dep_receipt, support_pins, support_dir, *, reader=None):
    """Call only from a separately authorized build after validating input receipts."""
    reader = reader or (lambda path: Path(path).read_bytes())
    expected = preflight(plan, dep_receipt, reader)
    if set(support_pins) != set(SUPPORT):
        raise ValueError("support header census differs")
    originals = dep_receipt["dependency_sha256"]
    writes = {}
    for alias, target in ORIGINAL_ALIASES.items():
        if (alias not in originals or posixpath.normpath(alias) != target
                or not Path(alias).samefile(target)):
            raise ValueError("pinned compiler alias is not the original file")
    bridge, = [row["original_source"] for row in plan["compiles"]
               if row["source_derivation"] == "prefix_bridge"]
    for name, pin in originals.items():
        data = reader(name)
        if digest(data) != pin:
            raise ValueError(f"effective dependency bytes differ: {name}")
        dest = original_mirror(name)
        if dest == name:
            continue
        if PurePosixPath(name).name in HEADERS:
            data = add_access(data)
        elif name == bridge:
            data = derive_bridge(data)
        if dest in writes and writes[dest] != data:
            raise ValueError("normalized dependency collision")
        writes[dest] = data

    parent = PurePosixPath(mirror(bridge)).parent
    for name in SUPPORT:
        data = (Path(support_dir) / name).read_bytes()
        if digest(data) != support_pins[name] or str(parent / name) in writes:
            raise ValueError("support bytes differ or collide")
        writes[str(parent / name)] = data
    root = Path(str(DEST))
    if any(not PurePosixPath(path).is_relative_to(MIRROR) for path in writes):
        raise ValueError("mirror destination escaped build directory")
    root.mkdir()
    (root / "obj").mkdir()
    for path, data in writes.items():
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(data)
    for alias, target in MIRRORED_ALIASES.items():
        # Retain the intermediate directory used by the compiler's include traversal.
        prefix = Path(alias.split("/../", 1)[0])
        if not prefix.is_relative_to(MIRROR) or not Path(target).is_relative_to(MIRROR):
            raise ValueError("unexpected compiler alias outside the mirror")
        prefix.mkdir(parents=True, exist_ok=True)
        if not Path(alias).samefile(target):
            raise ValueError("candidate compiler alias does not resolve to the mirrored file")
    return {"expected_dependencies_by_object": expected,
            "mirrored_sha256": {path: digest(data) for path, data in writes.items()}}


def verify_dependencies(raw, output, expected):
    actual = {MIRRORED_ALIASES.get(path, path)
              for path in dependencies(raw, output, MIRRORED_ALIASES)}
    want = set(expected)
    if actual != want:
        raise ValueError(f"compiler dependency closure differs: "
                         f"{len(want - actual)} missing, {len(actual - want)} unexpected")
    return len(actual)
