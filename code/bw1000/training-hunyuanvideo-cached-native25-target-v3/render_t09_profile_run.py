"""Derive a one-run profile package from the frozen successful T09 controller."""

from hashlib import sha256
import json
import re


BASE_MANIFEST_SHA = "aff3258198af45367e9d4b260b4052c58e42a4e8490b44a0fa936ff971c3b314"
EXACT = ("assigned_kfd.py", "commands.py", "COMPATIBILITY.json", "contract.py", "controller.py", "flash_compat.py",
         "framework_binding.py", "host_entry.py", "launch.py", "NATIVE_BINDING.json", "native_worker.py",
         "ORIGINAL_LAUNCH.json", "review.py", "runtime_pci.py", "slurm_accounting.py", "snapshot.py")
EXTRAS = ("stage_t09_phase_framework.py", "render_t09_phase_profile.py",
          "hunyuanvideo_phase_profile.py", "profile_run_review.py")


def patch(raw, old, new):
    before = old.encode()
    if raw.count(before) != 1:
        raise ValueError("Exact original launcher source anchor differs")
    return raw.replace(before, new.encode(), 1)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def render(base_manifest, base, extra, scope):
    if sha256(base_manifest).hexdigest() != BASE_MANIFEST_SHA or set(base) != set(EXACT) | {"CLAIM.json"}:
        raise ValueError("Frozen original controller provenance differs")
    originals = json.loads(base_manifest)["files"]
    if any(pin(raw) != originals[name] for name, raw in base.items()) or set(extra) != set(EXTRAS):
        raise ValueError("Original or added source inventory differs")
    for key, regex in {
        "label": r"[A-Za-z0-9-]{15,80}", "request": r"[0-9a-f]{32}",
        "parent": r"[0-9]{6,10}", "node": r"[A-Za-z0-9-]{3,24}",
        "weight_receipt_node": r"[A-Za-z0-9-]{3,24}", "profile_source_label": r"[A-Za-z0-9-]{15,80}",
        "profile_inventory_SHA256": r"[0-9a-f]{64}",
    }.items():
        if re.fullmatch(regex, scope[key]) is None:
            raise ValueError("Invalid privately bound execution scope")
    host_mount_device = scope["host_mount_device"]
    if type(host_mount_device) is not int or not 0 <= host_mount_device < 2**64:
        raise ValueError("Invalid observed HOST mount device")

    # Resolve the old scope literals from its pinned claim, so no live topology is in the renderer.
    previous = json.loads(base["CLAIM.json"])
    contract = base["contract.py"]
    for field, key in (("LABEL", "label"), ("REQUEST", "request_id")):
        value = scope["request" if key == "request_id" else key]
        contract = patch(contract, f'{field} = "{previous[key]}"', f'{field} = {json.dumps(value)}')
    contract = patch(contract,
        f'PARENT, NODE = "{previous["parent"]}", "{previous["node"]}"',
        f'PARENT, NODE = {json.dumps(scope["parent"])}, {json.dumps(scope["node"])}\n'
        f'WEIGHT_RECEIPT_NODE = {json.dumps(scope["weight_receipt_node"])}\n'
        f'PROFILE_SOURCE_LABEL = {json.dumps(scope["profile_source_label"])}\n'
        f'PROFILE_INVENTORY_SHA256 = {json.dumps(scope["profile_inventory_SHA256"])}')
    contract = patch(contract, 'and actual.get("node") == NODE',
                     'and actual.get("node") == WEIGHT_RECEIPT_NODE')
    # Rebind only the HOST mount-local device number. Preserve inode, size, timestamps and compute/full-SHA evidence.
    contract = patch(contract, 'HOST_MODEL_STAMP = [54, *MODEL_STAMP[1:]]',
                     f'HOST_MODEL_STAMP = [{host_mount_device}, *MODEL_STAMP[1:]]')

    commands = patch(base["commands.py"],
        'exec python3 -B -u "$@"',
        'export HUNYUAN_T09_PHASE_PROFILE=6,7\nexec python3 -B -u "$@"')
    worker = patch(base["native_worker.py"], "from framework_binding import verify_framework",
                   "from stage_t09_phase_framework import verify_staged\n"
                   "from contract import ROOT, PROFILE_SOURCE_LABEL, PROFILE_INVENTORY_SHA256")
    worker = patch(worker, "    framework = verify_framework(Path(FRAMEWORK), Path(FRAMEWORK_MANIFEST))",
                   '    profile_root = Path(ROOT) / "results" / PROFILE_SOURCE_LABEL / "framework"\n'
                   '    observer = (RUN / "src/hunyuanvideo_phase_profile.py").read_bytes()\n'
                   '    framework = verify_staged(Path(FRAMEWORK), profile_root, Path(FRAMEWORK_MANIFEST), observer)\n'
                   '    need(framework["inventory_SHA256"] == PROFILE_INVENTORY_SHA256, "Profile tree differs")')
    worker = patch(worker, 'str(FRAMEWORK / "tools/hygon_training_recipe.py")',
                   'str(profile_root / "tools/hygon_training_recipe.py")')
    controller = patch(base["controller.py"], "from review import COLUMNS, native, post",
                       "from review import COLUMNS, native, post\nfrom profile_run_review import read_phases")
    controller = patch(controller, '        result["native"] = native(RUN / "outputs/hunyuanvideo-1.5/train" / LABEL)',
                       '        output = RUN / "outputs/hunyuanvideo-1.5/train" / LABEL\n'
                       '        result["native"] = native(output)\n'
                       '        result["phase_profile"] = read_phases(output)')
    controller = patch(controller, 'and all(item["passed"] for item in receipts.values()))',
                       'and result["phase_profile"]["passed"]\n'
                       '                            and all(item["passed"] for item in receipts.values()))')

    claim = dict(previous, label=scope["label"], request_id=scope["request"], parent=scope["parent"],
                 node=scope["node"], state="SCOPED_NATIVE_PHASE_PROFILE",
                 recipe_scope="Same native 25 updates with only steps 6 and 7 profiled")
    selected = {name: raw for name, raw in base.items() if name != "CLAIM.json"}
    selected.update(extra, **{"contract.py": contract, "commands.py": commands, "native_worker.py": worker,
                             "controller.py": controller, "CLAIM.json": json.dumps(claim, indent=2).encode()})
    for name, raw in selected.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    selected["SOURCE_FILES.json"] = json.dumps({"files": {name: pin(raw) for name, raw in selected.items()}},
                                                indent=2).encode()
    return selected
