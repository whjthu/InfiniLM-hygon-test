"""Use the audited passive host PCI/KFD helper for this exact new scope."""
import os
from pathlib import Path
import re
import sys
from contract import LABEL, NODE, PARENT, RUN, load, need, save
from placement import authentic_step
from framework_mount import BIND_ENV_KEYS
import snapshot as helper
RUN = Path(RUN)


def main():
    phase = sys.argv[1]
    need(phase in ("pre", "gpu-gate", "post"), "Unknown host phase")
    helper.NODES = {NODE}
    helper.STEP = re.compile(re.escape(PARENT) + r"\.\d+")
    scheduler_phase="gpu" if phase=="gpu-gate" else phase
    handle=authentic_step(scheduler_phase,LABEL+"-"+scheduler_phase)
    row = helper.snapshot(False)
    route,_=load(RUN/"physical-route.json",16384)
    need(row["step"]==handle and row["boot_id"]==route["boot_id"] and row["pci"]==route["pci"],
         "T09_SAME_LIVE_ROUTE_BOOT_AND_EIGHT_PCI")
    row.update(actual_numeric_cgroup=True,actual_live_TRES_verified=True)
    need(row["kfd_holder_count"] == 0, "Host device holders are present or inventory incomplete")
    if phase != "pre":
        import json
        before = json.loads((RUN / "pre" / (NODE + ".json")).read_bytes())
        need(before["boot_id"] == row["boot_id"] and before["pci"] == row["pci"],
             "Independent host inventory differs")
        if phase == "gpu-gate":
            need(0 <= row["monotonic_ns"] - before["monotonic_ns"] <= 90_000_000_000,
                 "Pre-admission snapshot is stale")
    if phase == "gpu-gate":
        need(len(sys.argv) > 2 and Path(sys.argv[2]).is_absolute(), "GPU gate has no exact native command")
        row["environment_bind_override_present"]={name:bool(os.environ.get(name)) for name in BIND_ENV_KEYS}
        try:
            os.chdir("/")
            cwd=os.getcwd()
            row["host_getcwd"]="/" if cwd=="/" else "OTHER"
            need(cwd=="/","HOST_GETCWD_ROOT_BEFORE_EXEC")
        except (OSError, ValueError) as error:
            row["mount_context_error"]=dict(error_type=type(error).__name__,errno=getattr(error,"errno",None))
            save(RUN / phase / (NODE + ".json"), row)
            raise
    save(RUN / phase / (NODE + ".json"), row)
    if phase == "gpu-gate":
        os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
