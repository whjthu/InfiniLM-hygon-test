"""Recompile one text-model TU using the pinned 981/987/622 build lineage."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import socket
import subprocess
import time


ROOT = Path(__file__).resolve().parent
LABEL = "Kimi461-output-residual-tile64-CPU-v1"
OUTPUT = Path("/refactor/candidates") / LABEL
ALLOWED_CLOCK_SKEW = 2.0


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pin(entry):
    path = Path(entry["path"])
    need(checksum(path) == entry["SHA256"], "pinned original receipt differs")
    return json.loads(path.read_bytes())


def replace_once(argv, old, new):
    need(argv.count(str(old)) == 1, "original argv member missing or duplicated")
    argv[argv.index(str(old))] = str(new)


def link_objects(argv):
    objects = [word for word in argv if word.endswith(".o")]
    need(len(objects) == len(set(objects)) == 34, "link does not contain 34 distinct objects")
    return set(objects)


def lineage(compiled, linked, resident):
    """Rebuild the historical object substitutions without choosing a new linker recipe."""
    text = compiled["compiled"]["text_model"]
    obj = text["object"] if "object" in text else next(
        obj for obj, digest in compiled["all_linked_object_SHA256"].items()
        if PurePosixPath(obj).name == "text_model.o" and digest == text["object_SHA256"]
    )
    need(compiled["all_linked_object_SHA256"][obj] == text["object_SHA256"],
         "CPU981 text descriptor differs from the linked object")
    resident_objs = resident["all_linked_object_SHA256"]
    linked_objs = linked["all_linked_object_SHA256"]
    link, = [list(argv) for argv in linked["commands"] if "-shared" in argv]
    need(link_objects(link) == set(linked_objs), "CPU987 link and object receipt differ")
    old_worker = linked["original985_metadata"]["worker"]["object"]["path"]
    new_worker = str(PurePosixPath(resident["native"]).parent / "worker.o")
    need(old_worker in linked_objs and new_worker not in linked_objs and new_worker in resident_objs,
         "historical worker substitution is absent or ambiguous")
    need(resident_objs == {k: v for k, v in linked_objs.items() if k != old_worker}
         | {new_worker: resident_objs[new_worker]}, "CPU622 did not only replace the bound worker")
    need(resident_objs.get(obj) == linked_objs.get(obj) == text["object_SHA256"],
         "CPU981 text object no longer belongs to the baseline")
    replace_once(link, old_worker, new_worker)
    replace_once(link, linked["native"], resident["native"])
    need(link_objects(link) == set(resident_objs), "reconstructed CPU622 link differs from its object map")
    return text, obj, link


def private_link(original, old_object, new_object, old_library, new_library, map_path):
    argv = list(original)
    replace_once(argv, old_object, new_object)
    replace_once(argv, old_library, new_library)
    map_flag, = [word for word in argv if word.startswith("-Wl,-Map,")]
    replace_once(argv, map_flag, f"-Wl,-Map,{map_path}")
    return argv


def compile_commands(original, orig, old_object, work, source):
    cmd = list(original)
    need(cmd.count("-c") == cmd.count("-MD") == cmd.count("-MF") == cmd.count("-o") == 1
         and "-M" not in cmd and "-MMD" not in cmd
         and cmd[cmd.index("-c")+1] == orig and cmd[cmd.index("-o")+1] == old_object,
         "CPU981 text compile argv shape differs")
    obj, before, after = [str(work / name) for name in ("text_model.o", "text_model.before.d", "text_model.d")]
    old_md = cmd[cmd.index("-MF")+1]
    replace_once(cmd, orig, source)
    replace_once(cmd, old_object, obj)
    replace_once(cmd, old_md, after)
    pre = list(cmd)
    replace_once(pre, "-MD", "-M")
    replace_once(pre, after, before)
    at = pre.index("-o")
    del pre[at:at+2]
    pre.remove("-c")
    pre[1:1] = ["-MT", obj]
    return cmd, pre, Path(obj), Path(before), Path(after), Path(old_md)


def dep_paths(path, target):
    lhs, colon, rhs = path.read_text().replace("\\\n", " ").partition(": ")
    need(bool(colon) and shlex.split(lhs) == [str(target)], "make dependency target differs")
    deps = shlex.split(rhs)
    need(bool(deps) and all(PurePosixPath(p).is_absolute() for p in deps), "dependency has an unbound path")
    return set(deps)


def close_dependencies(old, old_source, new_source, header, actual):
    need(old_source in old and str(new_source) not in old and str(header) not in old,
         "original dependency source or candidate path differs")
    expected = (set(old) - {old_source}) | {str(new_source), str(header)}
    need(actual == expected, "text_model dependencies differ beyond the source and added header")
    return expected


def dynamic_identity(text):
    need("SYMBOLIC" in text, "missing original SYMBOLIC flag")
    return re.findall(r"\((NEEDED|RPATH|RUNPATH|SONAME)\).*?\[(.*?)\]", text)


def exports(text):
    return {parts[-1]: parts[-2] for line in text.splitlines()
            if len(parts := line.split()) >= 3 and parts[-2] in ("T", "D", "B", "R")}


def providers(text):
    need("not found" not in text and "undefined symbol" not in text, "unresolved ELF dependency")
    files = set()
    for line in text.splitlines():
        tokens = line.split()
        item = tokens[2] if len(tokens) >= 3 and tokens[1] == "=>" else (tokens[0] if tokens else "")
        if item.startswith("/"):
            files.add(str(Path(item).resolve(strict=True)))
    need(bool(files), "loader reported no concrete dependency paths")
    return files


def budget(epoch, deadline, cap, now=None):
    now = time.time() if now is None else now
    need(all(math.isfinite(value) for value in (epoch, deadline, cap, now))
         and epoch > 0 and deadline == epoch + 260 and epoch - ALLOWED_CLOCK_SKEW <= now and cap > 0,
         "invalid inherited build epoch")
    left = min(deadline - now - 5, cap)
    if left <= 0:
        raise TimeoutError("expired original text-model build deadline")
    return left


def pins_for(root, manifest_sha):
    path = root / "SOURCE_FILES.json"
    need(checksum(path) == manifest_sha, "source manifest differs")
    manifest = json.loads(path.read_bytes())["files"]
    need(set(manifest) == {"CLAIM.json", "CPU_outer.sh", "CPU_inside.sh", "CPU_build.py",
                           "launch_on_host.py", "kimi_k3_text_model.cpp", "output_residual_tiles.hpp"}
         and all(Path(name).name == name for name in manifest), "invalid package paths")
    for name, h in manifest.items():
        need(checksum(root/name) == h, "source package differs")
    claim = json.loads((root / "CLAIM.json").read_bytes())
    need(claim["label"] == LABEL and claim["GPU_GO"] is False and claim["Model_GO"] is False
         and claim["actual_trial_limit"] == 1, "wrong bounded CPU claim")
    return claim


def main():
    claim = pins_for(ROOT, os.environ["KIMI_TILE_MANIFEST_SHA256"])
    epoch = float(os.environ["KIMI_TILE_EPOCH"])
    deadline = float(os.environ["KIMI_TILE_INNER_DEADLINE"])
    budget(epoch, deadline, 260)
    need(socket.gethostname().split(".")[0] == claim["node"]
         and os.environ["SLURM_JOB_ID"] == claim["parent"]
         and os.environ["SLURM_STEP_ID"].isdigit(), "wrong exact CPU resource")
    need(len(os.sched_getaffinity(0)) >= claim["cpu_cores"]
         and all(os.environ.get(name) == "" for name in
                 ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"))
         and not any(os.environ.get(name) for name in ("LD_PRELOAD", "LD_AUDIT")),
         "CPU step environment differs")
    kfd = Path("/sys/class/kfd/kfd/proc")
    need(kfd.is_dir() and not any(kfd.iterdir()), "fresh empty or unavailable node device ownership")

    OUTPUT.mkdir(mode=0o700)
    resolved = dict(actual_CPU=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"], epoch=epoch,
                    deadline=deadline, source_manifest_SHA256=os.environ["KIMI_TILE_MANIFEST_SHA256"],
                    completed=False, passed=False, planned_new_host_TU=1, planned_reused_objects=33, explicit_GPU=0,
                    explicit_native_dlopen=False, checkpoint_read=False, commands=[])

    def run(name, argv, cap):
        left = budget(epoch, deadline, cap)
        resolved["phase"] = name
        resolved["commands"].append(argv)
        log = OUTPUT / (name + ".log")
        with log.open("xb") as f:
            subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=left, check=True)
        return log.read_text(errors="replace")

    try:
        compiled, linked, resident = (read_pin(claim["original_receipts"][k]) for k in ("981", "987", "622"))
        need([r["actual_CPU"] for r in (compiled, linked, resident)] == ["910407.981", "910407.987", "910380.622"]
             and all(r["passed"] and r["completed"] and not r["GPU"] and not r["Native_dlopen"]
                     for r in (compiled, linked, resident)),
             "pinned producer records do not match the measured build lineage")
        need(resident["native_SHA256"] == claim["original_native_SHA256"] == checksum(resident["native"]),
             "original resident library differs")
        need(linked["native_SHA256"] == checksum(linked["native"]), "CPU987 original library differs")
        need(resident["actual_dependency_library_SHA256"] == linked["actual_dependency_library_SHA256"],
             "CPU622 runtime dependency map differs")
        for lib in resident["actual_dependency_library_SHA256"].values():
            need(checksum(lib["path"]) == lib["SHA256"], "pinned SDK or runtime library differs")
        for path, h in resident["all_linked_object_SHA256"].items():
            need(checksum(path) == h, "resident object differs")
        text, old_obj, link = lineage(compiled, linked, resident)
        compiler = Path(shutil.which("g++")).resolve(strict=True)
        cc = linked["compiler_current_container"]
        need(compiler == Path(cc["canonical"]) and checksum(compiler) == cc["SHA256"],
             "selected container compiler differs from CPU987 identity")
        old_source = text["source"]
        need(checksum(old_source) == text["source_SHA256"] == claim["original_text_SHA256"],
             "pinned CPU981 text source differs")
        origin, = [argv for argv in compiled["commands"]
                   if "-c" in argv and argv[argv.index("-c")+1] == old_source]
        source, header = ROOT/"kimi_k3_text_model.cpp", ROOT/"output_residual_tiles.hpp"
        compile, pre, obj, before, after, old_md = compile_commands(origin, old_source, old_obj, OUTPUT, source)
        old_md_paths = text["all_new_MD"]
        need(text["MD_before_after_identical"] and dep_paths(old_md, old_obj) == set(old_md_paths),
             "pinned CPU981 original MD differs")
        for path, h in old_md_paths.items():
            need(checksum(path) == h, "pinned original header differs")
        compile[0] = pre[0] = link[0] = str(compiler)
        run("fresh-M", pre, 40)
        expected = close_dependencies(old_md_paths, old_source, source, header, dep_paths(before, obj))
        snapshot = {p: checksum(p) for p in sorted(expected)}
        need(all(snapshot[p] == h for p, h in old_md_paths.items() if p != old_source),
             "fresh M transferred a changed dependency")
        run("single-HOST-text-model", compile, 115)
        need(dep_paths(after, obj) == expected and all(checksum(p) == h for p, h in snapshot.items()),
             "compiled MD or dependency bytes differ from preprocessor closure")
        candidate = OUTPUT / Path(resident["native"]).name
        link = private_link(link, old_obj, obj, resident["native"], candidate, OUTPUT/"link.map")
        unchanged = {p: h for p, h in resident["all_linked_object_SHA256"].items() if p != old_obj}
        need(link_objects(link) == set(unchanged) | {str(obj)} and len(unchanged) == 33,
             "candidate changed more than one object")
        run("one-object-private-link", link, 50)
        old_exports = exports(run("original-exports", ["/usr/bin/nm", "-D", "--defined-only", resident["native"]], 12))
        new_exports = exports(run("candidate-exports", ["/usr/bin/nm", "-D", "--defined-only", str(candidate)], 12))
        need(bool(old_exports) and old_exports == new_exports, "strong ELF export inventory differs")
        old_dynamic = dynamic_identity(run("original-dynamic", ["/usr/bin/readelf", "-dW", resident["native"]], 12))
        new_dynamic = dynamic_identity(run("candidate-dynamic", ["/usr/bin/readelf", "-dW", str(candidate)], 12))
        need(old_dynamic == new_dynamic, "ELF dynamic bindings differ")
        old_providers = providers(run("original-relocations", ["/usr/bin/ldd", "-r", resident["native"]], 18))
        new_providers = providers(run("candidate-relocations", ["/usr/bin/ldd", "-r", str(candidate)], 18))
        need(old_providers == new_providers, "passive loader provider paths changed")
        need(all(checksum(p) == h for p, h in unchanged.items()), "reused object changed during link")
        resolved.update(passed=True, built_host_TUs=1, reused_objects=unchanged,
                        candidate=str(candidate), candidate_SHA256=checksum(candidate),
                        actual_object_SHA256=checksum(obj), exact_new_MD=snapshot,
                        old_strong_export_count=len(old_exports),
                        link_map_SHA256=checksum(OUTPUT/"link.map"),
                        actual_passive_providers={p: checksum(p) for p in sorted(new_providers)})
    except Exception as exc:
        resolved.update(error_type=type(exc).__name__, reason=str(exc)[:240])
        raise
    finally:
        resolved["completed"] = True
        with (OUTPUT/"CPU-result.json").open("x", encoding="utf-8") as f:
            json.dump(resolved, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
