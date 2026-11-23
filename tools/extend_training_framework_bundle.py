"""Build a portable common training bundle with the registered Z-Image diagnostic."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from build_training_framework_bundle import blob_identity, bounded, require, validated_manifest
from training_framework_identity import manifest_tree


BASE_COMMIT = "e743f45635d582d0286bb1a6de86cd16beb0580a"
BASE_BRANCH = "refs/heads/framework-dp2-alignment"
BASE_BUNDLE = "2b94b56c8e244d3816767add4563ec587b94c06f52eff25a8a1ad0403474e17f"
BASE_MANIFEST = "9459dd77d647c8a57b82f4fb4fab03d7f2182c0710906de42ce90e734c64288e"
BRANCH = "refs/heads/framework-dp2-zimage-alignment"
OVERLAY = "code/bw1000/training-common-zimage-fixed-first-overlay-v1"
RECORDED = "code/bw1000/training-zimage-coco20-recorded-reference-v2"
NEW_MODULE = "generation/src/gencuda/zimage_fixed_reference.py"
RECORDED_MODULE = "d5fed2197a8936f76a115a6934af601a1186f8106074600f24fee4ce6b98d6a7"
CHANGES = {
    "generation/src/gencuda/zimage_first_update.py": "eb7a72e108ab0f992904289ef50e6898a84a9314b12f7c33df3383bc78b40451",
    "generation/src/gencuda/zimage_native_record.py": "34a639fb7d8e598a5db1306e07789721ab26b71887e5b4b819c55bd83893110a",
    "tools/hygon_training_recipe.py": "b218b9a6c1b5b80aaf6df5a73006b06c1529a30a77b1e4ed7b6a8a1d7a9b4f96",
    "tools/hygon_zimage_native_record.py": "fcd31dda1e24995ae291966c690bfa18d48f552511fd5e2bf0b987592df81b68",
}


def pinned(path, limit, expected=None):
    with path.open("rb") as stream:
        raw = bounded(stream, limit)
    if expected:
        require(sha256(raw).hexdigest() == expected, f"Frozen source changed: {path.name}")
    return raw


def verify_entry(checkout):
    command = [sys.executable, "-B", "-s", str(checkout / "tools/hygon_training_recipe.py")]
    help_text = subprocess.check_output(command + ["T07.sft", "--fixed-first-update", "--help"], timeout=25)
    require(b"--reference-sha256" in help_text and b"--binding" in help_text, "Registered fixed input entry differs")
    invalid = subprocess.run(command + ["T07.sft", "--fixed-first-update", "--binding", "relative",
                            "--reference", str(checkout), "--reference-sha256", "0"*64, "--output", str(checkout),
                            "--epoch", "1", "--deadline", "2"], capture_output=True, timeout=25)
    require(invalid.returncode == 2 and b"absolute paths" in invalid.stderr, "Invalid binding went past the CLI")

    # The final call boundary is exercised from the imported tree. The stub
    # verifies the selected production route without performing a hardware run.
    program = """
import json, pathlib, sys, tempfile
root = pathlib.Path(sys.argv[1])
sys.path.insert(0,str(root/'tools'))
import hygon_training_recipe as recipe
import hygon_zimage_native_record as native
calls = []
def fake(binding, output, **options):
    calls.append((binding, output, options))
    return dict(passed=True,mode='first-fixed',actual_optimizer_updates=1)
native.run = fake
with tempfile.TemporaryDirectory() as folder:
    base = pathlib.Path(folder).resolve()
    binding = base/'binding.json'
    binding.write_text(json.dumps(dict(case='T07.sft',platform='hip')),encoding='utf8')
    args = ['T07.sft','--fixed-first-update','--binding',str(binding),'--reference',str(base),
            '--reference-sha256','a'*64,'--output',str(base/'out'),'--epoch','100','--deadline','160']
    assert recipe.main(args,generation_accelerator='hip') == 0 and len(calls) == 1
    record, output, kwargs = calls[0]
    assert record['case'] == 'T07.sft' and output == base/'out'
    assert kwargs == dict(mode='first-fixed',epoch=100.,deadline=160.,
                          first_reference=base,first_reference_sha256='a'*64)
    try:
        recipe.main(args,generation_accelerator='cuda')
    except SystemExit as error:
        assert error.code == 2 and len(calls) == 1
    else:
        raise AssertionError("Registry allowed a cross-accelerator binding")
"""
    subprocess.run([sys.executable, "-B", "-s", "-c", program, str(checkout)], timeout=25,
                   capture_output=True, check=True)
    return {"fixed_help": True, "relative_binding_rejected": True,
            "native_argument_routing_verified": True, "wrong_platform_rejected": True}


def create(project, manifest, prior_bundle, destination):
    inventory = validated_manifest(pinned(manifest, 512 << 10, BASE_MANIFEST))
    pinned(prior_bundle, 8 << 20, BASE_BUNDLE)
    files = dict(inventory["files"])
    replacements = {}
    for name, expected in CHANGES.items():
        require(files[name]["SHA256"] == expected, f"Previous member differs: {name}")
        replacements[name] = pinned(project / OVERLAY / "framework-delta" / name, 128 << 10)
    require(NEW_MODULE not in files, "The previous bundle already includes the new member")
    replacements[NEW_MODULE] = pinned(project / RECORDED / "framework-delta" / NEW_MODULE,
                                      128 << 10, RECORDED_MODULE)
    for name, raw in replacements.items():
        compile(raw, name, "exec")
        files[name] = {"mode": files.get(name, {"mode": 0o644})["mode"], "bytes": len(raw),
                       "SHA256": sha256(raw).hexdigest(), "git_blob": blob_identity(raw)}
    expected_tree = manifest_tree(files)

    destination.mkdir()
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS="1")
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)

    def run(argv, data=None):
        return subprocess.run(argv, env=env, input=data, capture_output=True, timeout=45, check=True).stdout

    source = destination / "source.git"
    verifier = destination / "verified.git"
    bundle = destination / "common-training-dp2-zimage.bundle"
    run(["git", "init", "--bare", str(source)])
    work = ["git", "--git-dir", str(source)]
    run([*work, "fetch", str(prior_bundle), f"{BASE_BRANCH}:{BASE_BRANCH}"])
    require(run([*work, "rev-parse", BASE_BRANCH]).decode().strip() == BASE_COMMIT, "Prior candidate differs")
    require(run([*work, "rev-parse", BASE_COMMIT + "^{tree}"]).decode().strip() == inventory["tree"],
            "Prior candidate tree differs")
    message = b"Add the registered Z-Image fixed-input first-update diagnostic\n"
    pack = bytearray(f"commit {BRANCH}\n".encode()
                     + b"committer whjthu <haojie0429@gmail.com> 1795425000 +0000\n"
                     + f"data {len(message)}\n".encode() + message + b"\n"
                     + f"from {BASE_COMMIT}\n".encode())
    for name, raw in sorted(replacements.items()):
        pack.extend(f'M {0o100000|files[name]["mode"]:o} inline {json.dumps(name)}\ndata {len(raw)}\n'.encode()
                    + raw + b"\n")
    pack.extend(b"\ndone\n")
    run([*work, "fast-import", "--quiet"], bytes(pack))
    run([*work, "bundle", "create", str(bundle), BRANCH])

    run(["git", "init", "--bare", str(verifier)])
    verify = ["git", "--git-dir", str(verifier)]
    run([*verify, "bundle", "verify", str(bundle)])
    run([*verify, "fetch", str(bundle), f"{BRANCH}:{BRANCH}"])
    commit = run([*verify, "rev-parse", BRANCH]).decode().strip()
    require(run([*verify, "rev-parse", commit + "^"]).decode().strip() == BASE_COMMIT, "Common parent changed")
    require(run([*verify, "rev-parse", commit + "^{tree}"]).decode().strip() == expected_tree,
            "Imported framework tree differs")
    diff = run([*verify, "diff", "--name-only", BASE_COMMIT, commit]).decode().splitlines()
    require(set(diff) == set(replacements), "Imported source scope differs")
    run([*verify, "fsck", "--strict", "--no-reflogs", commit])
    checkout = destination / "framework"
    checkout.mkdir()
    run([*verify, "--work-tree", str(checkout), "checkout", BRANCH, "--", "."])
    cases = json.loads(run([sys.executable, "-B", "-s", str(checkout / "tools/hygon_training_recipe.py"), "list"]))
    require(cases == inventory["registry_cases"] and len(cases) == 17, "Registered cases changed")
    masked_checks = verify_entry(checkout)
    result = {"candidate_commit": commit, "parent_commit": BASE_COMMIT, "tree": expected_tree, "branch": BRANCH,
              "files": len(files), "replaced_members": {n: files[n] for n in sorted(replacements)},
              "registry": cases, "bundle_bytes": bundle.stat().st_size, "bundle_sha256": sha256(pinned(bundle, 8 << 20)).hexdigest(),
              "fresh_import_verified": True, "registered_entry_checks": masked_checks, "executed_GPU_steps": 0}
    (destination / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2)+"\n", encoding="utf8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "manifest", "prior_bundle", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(create(*(p.resolve() for p in
                             (args.project,args.manifest,args.prior_bundle,args.destination))), sort_keys=True))
