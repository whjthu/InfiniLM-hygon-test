"""Validate the original CPU-v2 reader with local fixture bytes."""
from hashlib import sha256
import json
from pathlib import Path
import struct
import read_original_build as ns

base = Path(__file__).resolve().parents[1] / "CPU-v2"
epoch, end = 1793952000.25, 1793952300.25
intent = dict(publication=ns.PUBLICATION, source_manifest_SHA256=ns.SOURCE_MANIFEST, claim_SHA256=ns.CLAIM,
              once=True, GPU=False, Native_dlopen=False, HOST="recorded-host", epoch=epoch, deadline=end,
              argv=["srun","--jobid=910407","--job-name="+ns.LABEL,"-c4","--mem=8192M","--gres=none",
                    "a14r2n20","00:05:00","TUNING_CPU_CLAIM_SHA="+ns.CLAIM,
                    "TUNING_CPU_EPOCH="+repr(epoch),"TUNING_CPU_DEADLINE="+repr(end)])
launch = dict(HOST=intent["HOST"], epoch=epoch, deadline=end, GPU=False,
              owned_srun_PID=123, PGID=123, startticks="456", observed_at_epoch=epoch+1)
job = "910407.98765"
accounting = dict(rc=0, columns=ns.COLUMNS, stdout=f"{job}|{ns.LABEL}|COMPLETED|0:0|17")
q = dict(actual_CPU=job, completed=True, passed=True, CPU_only=True, new_HOST=1, new_GPU=0,
         Native_dlopen=False, Runtime=False, SDK_explicit_calls=0, GPU_job=False, weights=False,
         Torch_imported=False, Model_qualified=False, all_fresh_M_before_compile=True,
         all_fresh_M_equal_MD=True, unmatched_dependencies=[], project_maps=[],
         source_manifest_SHA256=ns.SOURCE_MANIFEST, claim_SHA256=ns.CLAIM,
         original_epoch=epoch, original_deadline=end, elapsed_s=15.0)
digest = lambda b: sha256(b).hexdigest()
probe_source = ns.PACKAGE+"/native-src/probe.cpp"
header = ns.PACKAGE+"/native-src/input_guard.hpp"
obj = ns.OUTPUT+"/probe.o"
md = ns.OUTPUT+"/probe.d"
before = ns.OUTPUT+"/probe.before.d"
receipt = ns.OUTPUT+"/CPU-result.json"
files = {}
for name in ("SOURCE_MANIFEST.json", "CPU_RESOURCE_CLAIM.json", "BUILD_PLAN.json"):
    files[ns.PACKAGE+"/"+name] = (base/name).read_bytes()
manifest = json.loads(files[ns.PACKAGE+"/SOURCE_MANIFEST.json"])
plan = json.loads(files[ns.PACKAGE+"/BUILD_PLAN.json"])
for name in manifest["files_SHA256"]:
    files[ns.PACKAGE+"/"+name] = (base/name).read_bytes()
deps = {probe_source:digest(files[probe_source]), header:digest(files[header]),
        "/refactor/pinned/deepseek_v4_c4_indexer.hpp":plan["class_header_SHA256"]}
files[md] = files[before] = f"{obj}: {' '.join(deps)}\n".encode()
files[obj] = b"synthetic object: provenance only, not ELF"
q["compiled"] = {"probe":dict(source=probe_source, source_SHA256=deps[probe_source],
                              object=obj, object_SHA256=digest(files[obj]), MD=md, MD_SHA256=digest(files[md]))}
q["fresh_dependencies_SHA256"] = {"probe":deps}
files[receipt] = json.dumps(q).encode()

item, = plan["links"]
native = item["required_NEEDED"]
def elf_fixture(names, symbolic=True, dep=native):
    strings = bytearray(b"\0")
    def add(name):
        offset = len(strings)
        strings.extend(name.encode()+b"\0")
        return offset
    names_offsets = [add(name) for name in names]
    dependency_offset = add(dep)
    symbols = bytes(24)+b"".join(struct.pack("<IBBHQQ", n, 0x12, 0, 1, 1, 1) for n in names_offsets)
    dynamic = struct.pack("<qQqQqQ", 1, dependency_offset, 16 if symbolic else 0, 0, 0, 0)
    result = bytearray(64)
    result[:6] = b"\x7fELF\x02\x01"
    s0 = len(result)
    result.extend(strings)
    s1 = len(result)
    result.extend(symbols)
    s2 = len(result)
    result.extend(dynamic)
    off = len(result)
    result.extend(bytes(64))
    for kind,start,size,link,entsize in (
        (3,s0,len(strings),0,0), (11,s1,len(symbols),1,24), (6,s2,len(dynamic),1,16),
    ):
        result.extend(struct.pack("<IIQQQQIIQQ",0,kind,0,0,start,size,link,0,0,entsize))
    struct.pack_into("<Q",result,40,off)
    struct.pack_into("<HH",result,58,64,4)
    return bytes(result)

library_path = ns.OUTPUT + "/" + item["output"]
files[library_path] = elf_fixture(item["required_exports"])
q["libraries"] = {"probe": dict(path=library_path, SHA256=digest(files[library_path]),
                               all_linked_object_SHA256={obj:digest(files[obj])}, NEEDED=[native],
                               all_transitive_resolved_library_SHA256=plan["dynamic_SHA256"])}
q["commands"] = [dict(name="probe-link",argv=["g++","-Wl,--no-undefined",obj,native,"-linfiniops"])]
q["logs"] = {}
for stream in ("stdout","stderr"):
    name = "probe-ldd."+stream
    filename = ns.OUTPUT+"/"+name
    files[filename] = ("\n".join("fixture => "+p+" (0x1234)" for p in plan["dynamic_SHA256"])
                       if stream == "stdout" else "").encode()
    q["logs"][name] = dict(path=filename, SHA256=digest(files[filename]), bytes=len(files[filename]))
files[receipt] = json.dumps(q).encode()
result = ns.read_build(intent,launch,accounting,job,digest(files[receipt]),files.__getitem__)
assert result["build_provenance_checked"] and result["passive_ELF_verified"]
assert result["transitive_count"] == 69 and len(result["exports"]) == 13 and not result["GPU_GO"]
for path in (library_path, ns.OUTPUT+"/probe-ldd.stdout", ns.OUTPUT+"/probe-ldd.stderr"):
    corrupt = dict(files)
    corrupt[path] += b"x"
    try:
        ns.read_build(intent,launch,accounting,job,digest(files[receipt]),corrupt.__getitem__)
    except ValueError:
        continue
    raise AssertionError("accepted corrupt linkage artifact")
failed = dict(accounting, stdout=accounting["stdout"].replace("COMPLETED|0:0","TIMEOUT|0:15"))
def unexpected_read(path):
    raise AssertionError("terminal failure must not read artifacts")
assert not ns.read_build(intent,launch,failed,job,None,unexpected_read)["build_provenance_checked"]
print(json.dumps(dict(full_reader=True, rejected_corrupt_linkage_artifacts=3, failed_terminal_artifact_reads=0)))
