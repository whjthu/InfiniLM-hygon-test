"""Compare the frozen ACCAD locomotion cohort and publisher WoodenMesh skeletons."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import struct
from zipfile import ZipFile

from audit_hymotion_accad_bvh import ARCHIVES, CHANNELS


SELECTION_SHA = "b9b4f430f9d9cb65cd80d021cf0e1d1bf853a770316e68f5a5848f2ce90cea03"
WOODEN_ASSETS = {
    "joint_names.json": (698, "f29139c07a3e8c160cb3512428aac6e272697ade096a7e682024c47424e064d1"),
    "kintree.bin": (208, "98a20fa3b53193790b63d9ac3a9c917f2f70fbe6e053dca495c75317ff4b756a"),
    "j_template.bin": (624, "4f488cc9d0b650a816f5a8eda49eda7bc796f490e9130a6e0dec5be137d7b929"),
}
WOODEN_TO_BVH = {
    "Pelvis": "Hips", "L_Hip": "LeftUpLeg", "R_Hip": "RightUpLeg",
    "Spine1": "ToSpine", "L_Knee": "LeftLeg", "R_Knee": "RightLeg",
    "Spine2": "Spine", "L_Ankle": "LeftFoot", "R_Ankle": "RightFoot",
    "Spine3": "Spine1", "L_Foot": "LeftToeBase", "R_Foot": "RightToeBase",
    "Neck": "Neck", "L_Collar": "LeftShoulder", "R_Collar": "RightShoulder",
    "Head": "Head", "L_Shoulder": "LeftArm", "R_Shoulder": "RightArm",
    "L_Elbow": "LeftForeArm", "R_Elbow": "RightForeArm",
    "L_Wrist": "LeftHand", "R_Wrist": "RightHand",
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def skeleton(stream):
    header = []
    for _ in range(256):
        raw = stream.readline(1025)
        if not raw or len(raw) > 1024:
            raise ValueError("missing or oversized BVH hierarchy")
        line = raw.decode("ascii").strip()
        if line == "MOTION":
            break
        if line:
            header.append(line)
    else:
        raise ValueError("BVH motion marker absent within header bound")
    lines = iter(header)
    if next(lines, "") != "HIERARCHY":
        raise ValueError("missing HIERARCHY marker")
    joints = []
    names = set()

    def parse(line, parent, depth):
        if depth > 24:
            raise ValueError("excessive BVH hierarchy depth")
        match = re.fullmatch(r"(ROOT|JOINT) (\w+)", line)
        end = line == "End Site"
        if (not end and match is None) or ((parent is None) != line.startswith("ROOT ")):
            raise ValueError("invalid BVH joint declaration")
        if next(lines, "") != "{":
            raise ValueError("missing BVH joint body")
        offset = next(lines, "").split()
        if len(offset) != 4 or offset[0] != "OFFSET":
            raise ValueError("missing BVH OFFSET")
        position = tuple(float(value) for value in offset[1:])
        if not all(isfinite(value) for value in position):
            raise ValueError("nonfinite BVH offset")
        if end:
            if next(lines, "") != "}":
                raise ValueError("unexpected End Site channels or children")
            return
        name = match[2]
        if name in names:
            raise ValueError("duplicate BVH joint name")
        names.add(name)
        channel = next(lines, "").split()
        width = 6 if parent is None else 3
        expected = CHANNELS if parent is None else {item for item in CHANNELS if item.endswith("rotation")}
        if channel[:2] != ["CHANNELS", str(width)] or len(channel[2:]) != width or set(channel[2:]) != expected:
            raise ValueError("invalid BVH channel layout")
        joints.append({"name": name, "parent": parent, "channels": channel[2:], "offset": position})
        for child in lines:
            if child == "}":
                return
            parse(child, name, depth + 1)
        raise ValueError("unclosed BVH joint")

    parse(next(lines, ""), None, 0)
    if next(lines, None) is not None:
        raise ValueError("additional BVH data after root")
    return joints


def inspect_wooden(path):
    blobs = {}
    for name, (size, pin) in WOODEN_ASSETS.items():
        with (path / name).open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size or sha256(raw).hexdigest() != pin:
            raise ValueError(f"publisher WoodenMesh bytes differ: {name}")
        blobs[name] = raw
    names = json.loads(blobs["joint_names.json"])
    parents = tuple(value for value, in struct.iter_unpack("<i", blobs["kintree.bin"]))
    positions = tuple(struct.iter_unpack("<fff", blobs["j_template.bin"]))
    if (not isinstance(names, list) or len(names) != len(parents) or len(names) != len(positions)
            or len(names) < 22 or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names) or parents[0] not in {-1, 0}
            or not all(0 <= parent < index for index, parent in enumerate(parents[1:], 1))
            or not all(isfinite(value) for position in positions for value in position)):
        raise ValueError("inconsistent publisher WoodenMesh names, kintree or rest positions")
    return {
        "assets": {name: {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()} for name, raw in blobs.items()},
        "total_joints": len(names),
        "target_joints": [{"name": name, "parent_index": parent} for name, parent in zip(names[:22], parents[:22])],
    }


def inspect_mapping(source, target):
    if set(WOODEN_TO_BVH) != {joint["name"] for joint in target}:
        raise ValueError("unexpected WoodenMesh target joint names")
    order = {joint["name"]: index for index, joint in enumerate(source)}
    source_names = [WOODEN_TO_BVH[joint["name"]] for joint in target]
    missing = [name for name in source_names if name not in order]
    if missing:
        return {"status": "MISSING_BVH_JOINTS", "missing_source_joints": missing}
    for joint, name in zip(target, source_names):
        parent = joint["parent_index"]
        if source[order[name]]["parent"] != (source_names[parent] if parent >= 0 else None):
            return {"status": "PARENT_GRAPH_DIFFERS", "first_bvh_joint": name}
    return {
        "status": "NAME_AND_PARENT_GRAPH_MATCH",
        "bvh_joint_indices_in_wooden_order": [order[name] for name in source_names],
    }


def audit(directory, wooden):
    raw = (directory / "locomotion-selected-85.private.json").read_bytes()
    if sha256(raw).hexdigest() != SELECTION_SHA:
        raise ValueError("locomotion candidate selection SHA differs")
    selection = json.loads(raw)["records"]
    if (len(selection) != 85 or len({(r["archive"], r["member"]) for r in selection}) != 85
            or Counter(r["split"] for r in selection) != Counter(training=67, development=18)):
        raise ValueError("locomotion candidate membership differs")
    target = inspect_wooden(wooden)
    cohorts = {}
    for name, (_, length, pin, split) in ARCHIVES.items():
        selected = [row for row in selection if row["archive"] == name]
        if not selected or any(row["split"] != split for row in selected):
            raise ValueError(f"unexpected cohort split: {name}")
        with (directory / name).open("rb") as stream:
            archive = stream.read(length + 1)
            if len(archive) != length or sha256(archive).hexdigest() != pin:
                raise ValueError(f"archive SHA or size differs: {name}")
            stream.seek(0)
            profiles = defaultdict(lambda: {"offset_sets": set(), "clips": 0})
            with ZipFile(stream) as source:
                for row in selected:
                    if source.getinfo(row["member"]).file_size != row["bytes"]:
                        raise ValueError(f"original member extent differs: {name}")
                    with source.open(row["member"]) as member:
                        joints = skeleton(member)
                    shape = [{key: value for key, value in joint.items() if key != "offset"} for joint in joints]
                    if sum(len(joint["channels"]) for joint in shape) != row["channel_count"]:
                        raise ValueError(f"BVH original channel count differs: {name}")
                    profile = profiles[digest(shape)]
                    profile["joints"] = shape
                    profile["clips"] += 1
                    profile["offset_sets"].add(digest(joints))
        for profile in profiles.values():
            profile["offset_sets"] = len(profile["offset_sets"])
            profile["mapping"] = inspect_mapping(profile["joints"], target["target_joints"])
        cohorts[name] = {"split": split, "clips": len(selected), "topologies": dict(profiles)}
    if sum(cohort["clips"] for cohort in cohorts.values()) != len(selection):
        raise ValueError("locomotion selection includes an unknown archive")
    return {
        "selection_SHA256": SELECTION_SHA,
        "cohorts": cohorts,
        "publisher_wooden": target,
        "retargeted_training_rows": 0,
        "blind_groups": 0,
        "GPU_steps": 0,
        "scope": "Header topology and publisher rest-skeleton identity only. No assumed BVH length units or coordinate transform.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("publisher_skeleton", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    doc = audit(args.candidate_directory, args.publisher_skeleton)
    data = (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"SHA256": sha256(data).hexdigest(), "bytes": len(data)}, sort_keys=True))


if __name__ == "__main__":
    main()
