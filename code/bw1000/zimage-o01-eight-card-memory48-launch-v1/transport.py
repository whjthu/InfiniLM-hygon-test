"""Use the already pinned project transport for the new HOST stage or one phase."""

from pathlib import Path
import sys
from types import ModuleType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render


PREVIOUS_SHA = "4dc598bc0420f0007084b330a465cbd8b37a93c4b05854e2453a130984d7cffe"


def adapter():
    path = render.OLD / "transport.py"
    selected = ModuleType("previous_o01_finite_transport")
    selected.__file__ = str(path)
    exec(compile(render.exact(path, PREVIOUS_SHA), str(path), "exec"), vars(selected))
    selected.STAGE = render.patch(selected.STAGE, render.OLD_LABEL, render.new.LABEL)
    selected.REQUEST = render.new.REQUEST
    installer = render.patch(selected.INSTALLER, render.OLD_REQUEST, render.new.REQUEST)
    installer = render.patch(installer, "'host_scope.py','node_entry.py','node_inventory.py','owner2108.json','pending.py','scheduler.py'}",
                             "'host_scope.py','node_entry.py','node_inventory.py','owner2108.json','pending.py','scheduler.py',"
                             "'identity_original.py','memory_link.py','memory_reader.py','pending_original.py'}")
    installer = render.patch(installer, "'mask_preflight.py','standby_worker.py','supervisor.py','witness.py'}",
                             "'mask_preflight.py','standby_worker.py','supervisor.py','witness.py','ram_trace.py'}")
    compile(installer, "memory48_installer", "exec")
    selected.INSTALLER = installer
    return selected


if __name__ == "__main__":
    raise SystemExit(adapter().main())
