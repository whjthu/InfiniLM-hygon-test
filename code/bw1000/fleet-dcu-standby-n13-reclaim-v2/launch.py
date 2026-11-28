"""Restore preemptible work on n13 using the previously qualified one-node guard."""

from hashlib import sha256
from pathlib import Path
import sys


PREVIOUS_SOURCE_SHA256 = "5082485c6521fdc0672eae5f6b36a0bb7e0d433d8a3ea28e995836cc7df46c83"
OBSERVED_RECEIPT_SHA256 = "97790934a21239439621c300e670236aa056c2090a19554fdefc7673c251d808"
REQUEST = "2685c925c9fe45c4b2b95d94411f3500"
PREDECESSOR = "1103606.4893"
PREDECESSOR_LABEL = "training80-dcu-standby-7db48c8e2a9b-f12r2n13"
CONTROL_STEP_PREFIXES = {
    "1103606.5124": "1c7767e9bfed",
    "1103606.5126": "191ba36fd8fb",
    "1103606.5128": "de2018011524",
    "1103606.5138": "d3b1d8cac542",
    "1103606.5153": "d969a0242be6",
    "1103606.5168": "185202cd25b8",
    "1103606.5183": "feba134d0153",
    "1103606.5195": "0f9c4292cde8",
    "1103606.5215": "879c5fe1eb35",
    "1103606.5222": "6f39f24f9363",
    "1103606.5224": "f09d0f9e0b86",
}


def main():
    publication, prior_source = sys.argv[1:]
    source = Path(prior_source)
    with source.open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    if not 0 < len(raw) <= 64 << 10 or sha256(raw).hexdigest() != PREVIOUS_SOURCE_SHA256:
        raise ValueError("Exact original one-node guard required")
    namespace = {"__name__": "qualified_n13_reclaim"}
    exec(compile(raw, str(source), "exec"), namespace)
    namespace.update(REQUEST=REQUEST, OLD=PREDECESSOR, OLD_LABEL=PREDECESSOR_LABEL, OBSERVER=None,
                     LABEL=f"training80-dcu-standby-{REQUEST[:12]}-f12r2n13",
                     CONTROLS={step: f"training80-standby-compute-{prefix}"
                               for step, prefix in CONTROL_STEP_PREFIXES.items()})
    print(namespace["json"].dumps(namespace["run"](publication), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
