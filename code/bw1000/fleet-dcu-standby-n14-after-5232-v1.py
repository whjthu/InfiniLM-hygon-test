"""Backfill the eight n14 DCUs returned by the completed LLaVA Val16 scope."""
from hashlib import sha256
import json
from pathlib import Path
import sys


PREVIOUS_SHA256 = "5082485c6521fdc0672eae5f6b36a0bb7e0d433d8a3ea28e995836cc7df46c83"
REQUEST = "8bc0692dbede42e38d21874ef7b61f6e"
PARENT, NODE = "1103606", "f13r1n14"
STANDBY = "1103606.5220"
STANDBY_LABEL = "training80-dcu-standby-879c5fe1eb35-f13r1n14"
MODEL_REQUEST = "7344da28bdbd49c59060db6595a3d692"
LABEL = "llava-clevr-5000-val16-v4-7344da28bdbd"
POST_SHA256 = "d525e9e55894c8a9c80335aa5601b167be713b813c2a7e3741b751d27892f71d"
OWNER_OUTCOME_SHA256 = "bde0bb3a4fa55f93dd6a289ccdb17f94b0826e8ffcf57845047a95e5925cf7b4"


def main():
    publication, previous = sys.argv[1:]
    p = Path(previous)
    with p.open("rb") as f:
        original = f.read(65537)
    if not 0 < len(original) <= 65536 or sha256(original).hexdigest() != PREVIOUS_SHA256:
        raise ValueError("One-node admission source differs")
    scope = {"__name__": "qualified_standby_admitter"}
    exec(compile(original, str(p), "exec"), scope)
    req = scope["require"]
    result = scope["RESULTS"] / LABEL
    post = json.loads(scope["read_source"](result / "post.json", POST_SHA256))
    outcome = json.loads(scope["read_source"](result / "controller-outcome.json", OWNER_OUTCOME_SHA256))
    req(post["request"] == outcome["request"] == MODEL_REQUEST and post["step"] == "1103606.5233"
        and post["node"] == NODE and post["role"] == "post" and post["port_clear"] is True
        and post["kfd_holder_count"] == post["owned_processes_remaining"] == 0 and outcome["cleanup_verified"] is True,
        "Exact prior model cleanup proof")
    pci = tuple(sorted(post["pci"]))
    req(len(pci) == len(set(pci)) == 8, "Complete project PCI roster")
    for step, role, resource in (("1103606.5232", "gpu", "cpu=8,gres/dcu=1,mem=62.50G,node=1"),
                                 ("1103606.5233", "post", "cpu=2,mem=2G,node=1")):
        [row] = scope["account"](step)
        req(row[1:4] == [LABEL + "-" + role, "COMPLETED", "0:0"] and row[5:7] == [NODE, resource]
            and row[8] not in ("", "Unknown"), "Exact terminal from the prior model role")
    scope.update(REQUEST=REQUEST, NODE=NODE, PARENT=PARENT, OLD=STANDBY, OLD_LABEL=STANDBY_LABEL,
                 PINNED_PCI=pci, PROTECTED={}, CONTROLS={}, OBSERVER=None,
                 LABEL=f"training80-dcu-standby-{REQUEST[:12]}-{NODE}")
    print(json.dumps(scope["run"](publication), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
