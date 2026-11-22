"""Use the exact finite executor receiver for the new reviewed HOST checkout."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_stage
import worker


OLD = worker.PRIOR_DIR / "stage_packets.py"
OLD_SHA = "9b8f53a037feee316572252a7bc3ce638e780975d5e99457ee9f948129956f55"


def programs(checkout, claim_sha, manifest_sha):
    worker.require(all(re.fullmatch(r"[0-9a-f]{64}", pin) for pin in (claim_sha, manifest_sha)),
                   "exact intended stage pins required")
    stage = prepare_stage.NEW
    checkout = Path(checkout).resolve(strict=True)
    claim_member = (stage / "SCOPE.json").as_posix()
    manifest_member = (stage / "STAGE_FILES.json").as_posix()
    claim = json.loads(worker.checked(checkout / claim_member, claim_sha, 16384))
    worker.require_request(claim["request"])
    manifest = json.loads(worker.checked(checkout / manifest_member, manifest_sha, 32768))
    count = len(manifest["files_SHA256"]) + 1
    worker.require(claim["label"] == worker.LABEL and claim["request"] == worker.REQUEST
                   and claim["stage_members"] == count - 1 and manifest["files_SHA256"][claim_member] == claim_sha,
                   "the new source transfer differs")
    text = worker.checked(OLD, OLD_SHA).decode()
    text = worker.once(text, "len(encoded) == 33", f"len(encoded) == {count}")
    text = worker.once(text, "len(content) != 33", f"len(content) != {count}")
    text = text.replace("V2_SOURCE_", "T512_V2_SOURCE_")
    scope = worker.execute(OLD, text)
    scope.update(LABEL=worker.LABEL, REQUEST=worker.REQUEST, CLAIM_SHA=claim_sha, MANIFEST_SHA=manifest_sha,
                 SCOPE=stage.as_posix(), CLAIM_MEMBER=claim_member, MANIFEST_MEMBER=manifest_member)
    return scope["programs"](checkout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--stage-manifest-sha256", required=True)
    args = parser.parse_args()
    files = programs(args.checkout, args.claim_sha256, args.stage_manifest_sha256)
    args.output.mkdir()
    for index, raw in enumerate(files):
        path = args.output / f"stage-{index:02}.py"
        with path.open("xb") as stream:
            stream.write(raw)
    print(json.dumps(dict(files=[sha256(raw).hexdigest() for raw in files], max_bytes=max(map(len, files)),
                          actual_new_Slurm_steps=0)))


if __name__ == "__main__":
    main()
