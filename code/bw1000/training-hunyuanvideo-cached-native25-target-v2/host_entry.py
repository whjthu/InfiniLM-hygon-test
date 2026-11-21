"""Use the audited passive host PCI/KFD helper for this exact new scope."""
import os
from pathlib import Path
import re
import sys
from contract import NODE, PARENT, RUN, need, save
import snapshot as helper
RUN = Path(RUN)


def main():
    phase = sys.argv[1]
    need(phase in ("pre", "gpu-gate", "post"), "Unknown host phase")
    helper.NODES = {NODE}
    helper.STEP = re.compile(re.escape(PARENT) + r"\.\d+")
    row = helper.snapshot(False)
    need(row["kfd_holder_count"] == 0, "Host device holders are present or inventory incomplete")
    if phase != "pre":
        import json
        before = json.loads((RUN / "pre" / (NODE + ".json")).read_bytes())
        need(before["boot_id"] == row["boot_id"] and before["pci"] == row["pci"],
             "Independent host inventory differs")
        if phase == "gpu-gate":
            need(0 <= row["monotonic_ns"] - before["monotonic_ns"] <= 90_000_000_000,
                 "Pre-admission snapshot is stale")
    save(RUN / phase / (NODE + ".json"), row)
    if phase == "gpu-gate":
        need(len(sys.argv) > 2, "GPU gate has no exact native command")
        os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
