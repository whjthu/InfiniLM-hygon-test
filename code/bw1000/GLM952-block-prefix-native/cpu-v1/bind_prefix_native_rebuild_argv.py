"""Derive GLM952 private prefix compile and link argv from the recorded commands."""

import argparse
import json
from pathlib import Path, PurePosixPath

from plan_prefix_native_rebuild import NATIVE_SHA, OBJECT_ROOT, RECEIPT_SHA, SNAPSHOT_SHA, checked

ARGV_SHA = "9ecf8517e2ed3c1dda5e895185765db91366ed14ca9e21dffe6f939c3b742e4d"
PARTITION_SHA = "11666cd4bf49e50bddf3c8c4bf8e6a314af330bd677d8261f8c15c71c6d52838"
DEST = PurePosixPath("/refactor/candidates/GLM952-block-prefix-native-v1")
MIRROR = DEST / "mirror"
BRIDGE = "modern_glm_bridge_observer_r131.cpp"
FLAGS = {"-std=c++17", "-O2", "-fPIC", "-D_GLIBCXX_USE_CXX11_ABI=1", "-pthread"}


def mirror(path):
    p = PurePosixPath(path)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("unexpected original path")
    if p.parts[1:3] in (("refactor", "candidates"), ("refactor", "incoming")):
        return str(MIRROR / p.relative_to("/refactor"))
    return path


def bind(partition, metadata, receipt):
    if partition["parent_Native_SHA256"] != receipt["library_SHA256"] or receipt["library_SHA256"] != NATIVE_SHA:
        raise ValueError("wrong original Native")
    original = receipt["final_objects_SHA256"]
    classes = {OBJECT_ROOT + row["object"]: row for row in partition["recompile"]}
    records = {row["object"]: row for row in metadata["rows"]}
    if (len(classes) != 8 or len(records) != 8 or records.keys() != classes.keys()
            or len(metadata["rows"]) != len(records) or len(original) != 29):
        raise ValueError("incomplete original compile record set")

    compiles, objects = [], {}
    for index, (target, info) in enumerate(records.items()):
        dependency = classes[target]
        if (info["exact_command_count"] != 1 or not info["receipt_completed"]
                or not info["MD_exact_target"] or not info["source_is_dependency"]
                or info["object_sha256"] != original[target]
                or original[target] != dependency["original_object_SHA256"]
                or info["MD_sha256"] != dependency["dependency_sha256"]):
            raise ValueError("original object, compile or dependency metadata disagree")
        argv = info["compile_argv"]
        source, md = info["source"], info["MD_path"]
        options = argv[1:-7]
        if (argv[0] != "/usr/bin/g++" or argv[-7:] != ["-MD", "-MF", md, "-c", source, "-o", target]
                or len([v for v in options if not v.startswith("-I")]) != len(FLAGS)
                or {v for v in options if not v.startswith("-I")} != FLAGS
                or PurePosixPath(source).name != dependency["source"]):
            raise ValueError("unexpected original compiler argument shape")

        adjusted = ["-I" + mirror(v[2:]) if v.startswith("-I") else v for v in options]
        candidate_source = mirror(source)
        if candidate_source == source:
            raise ValueError("source must be inside the candidate mirror")
        output = str(DEST / "obj" / f"{index:02d}.o")
        dependencies = str(DEST / "obj" / f"{index:02d}.d")
        original_headers = info["modified_class_headers"]
        if not original_headers or any(mirror(h) == h for h in original_headers):
            raise ValueError("unmapped class header")
        compiles.append(dict(
            original_object=target, original_source=source, source_sha256=info["source_sha256"],
            original_MD=md, original_MD_sha256=info["MD_sha256"], original_argv=argv,
            expected_rewritten_headers={h: mirror(h) for h in original_headers},
            source_derivation="prefix_bridge" if PurePosixPath(source).name == BRIDGE else "byte_exact_copy",
            preprocess_argv=[argv[0], *adjusted, "-M", "-MF", dependencies + ".pre",
                             "-MT", output, candidate_source],
            compile_argv=[argv[0], *adjusted, "-MD", "-MF", dependencies, "-c", candidate_source, "-o", output],
        ))
        objects[target] = output

    link, = [cmd for cmd in receipt["commands"] if "-shared" in cmd and "-o" in cmd]
    original_link_order = [arg for arg in link if arg.endswith(".o")]
    if (link[0] != "/usr/bin/g++" or link.count("-o") != 1 or "-Wl,--no-undefined" not in link
            or link[link.index("-o") + 1] != receipt["library"] or len(original_link_order) != len(original)
            or set(original_link_order) != original.keys()):
        raise ValueError("original link object inventory differs")
    new_link = [objects.get(arg, arg) for arg in link]
    new_link[new_link.index("-o") + 1] = str(DEST / "libGLM952_block_prefix_v1.so")
    return dict(
        build_authorized=False, destination=str(DEST), original_Native_SHA256=NATIVE_SHA,
        compile_count=len(compiles), retained_link_objects=len(original) - len(compiles),
        compiles=compiles, original_link_argv=link, link_argv=new_link,
        before_compile="Verify effective compiler/SDK, original object/source/MD bytes, then fresh -M "
                       "against all original dependencies mapped into the mirror, plus the three candidate "
                       "bridge support headers. Reject original class header paths and unresolved dependencies.",
        after_compile="Require exact -M/-MD equivalence; preserve all original strong and C ABI exports "
                      "with exactly the three private C ABI additions and original dynamic dependencies.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("partition", "command_metadata", "receipt_snapshot"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    snapshot = checked(args.receipt_snapshot.read_bytes(), SNAPSHOT_SHA)["files"]
    raw, = [r["rawtext"].encode() for p, r in snapshot.items()
            if p.endswith("/GLM-V-on-demand-Model-integration-r403-CPU/CPU-build.json")]
    report = bind(checked(args.partition.read_bytes(), PARTITION_SHA),
                  checked(args.command_metadata.read_bytes(), ARGV_SHA), checked(raw, RECEIPT_SHA))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
