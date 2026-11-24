"""Render a diagnostic-only FD audit overlay for a separately approved future scope."""

from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE = {
    "runtime/linux_owner.py": "5bb01fdd4a074a69b19fdfcb15c29fbbb071760f9df9854a5637c11713b83948",
    "runtime/supervisor.py": "b78c0ee935df8591a60da373b1275adb06aafad12c82ba4faaeaf14a03ec0567",
}


def replace(text, anchor, value):
    if text.count(anchor) != 1:
        raise ValueError("frozen code anchor differs")
    return text.replace(anchor, value, 1)


def instrument_owner(source):
    source = replace(source, "import signal\n", "import signal\nimport fd_permission\n")
    source = replace(source, "def audit_device_holders(standbys, expected_pcis):",
                     "def audit_device_holders(standbys, expected_pcis, *, step):")
    source = replace(source, "    render_holders = {}\n", """
    process_owners = {str(child.identity.pid): child.identity for child in standbys.values()}
    scan = fd_permission.Scan(PROC, os.getuid(), os.getpid(), step, process_owners, kfd, identity)
    render_holders = {}
""".lstrip("\n"))
    for anchor, replacement in (
        ("            if process.stat().st_uid != os.getuid():", "            if not scan.uid_matches(process):"),
        ('            for entry in (process / "fd").iterdir():', "            for entry in scan.fd_entries(process):"),
        ("                    target = os.readlink(entry)", "                    target = scan.fd_target(process, entry)"),
    ):
        source = replace(source, anchor, replacement)
    return source


def instrument_supervisor(source):
    source = replace(source, "import linux_owner as owned\n", "import linux_owner as owned\nimport fd_permission\n")
    source = replace(source, '"witness.py", "contract.py", "ram_trace.py"}',
                     '"witness.py", "contract.py", "ram_trace.py", "fd_permission.py"}')
    source = replace(source, """    def clean_target(self, siblings):
        return owned.audit_device_holders(siblings, GPU_SET)
""", """    def clean_target(self, siblings, boundary):
        try:
            return owned.audit_device_holders(siblings, GPU_SET, step=self.step)
        except fd_permission.HolderAccessDenied as error:
            self.results["holder_audit_denial"] = {"boundary": boundary, **error.diagnostic}
            raise
""")
    for boundary in ("pre_witness", "pre_restore"):
        anchor = f'    report["physical_{boundary}"] = step.clean_target(siblings)\n'
        source = replace(source, anchor, anchor.replace("(siblings)", f'(siblings, "{boundary}")'))
    return source


def overlay(members):
    """Return the three affected runtime members and require new package manifests upstream."""
    for key, pin in BASELINE.items():
        if sha256(members[key]).hexdigest() != pin:
            raise ValueError("frozen runtime bytes differ: " + key)
    new = {
        "runtime/linux_owner.py": instrument_owner(members["runtime/linux_owner.py"].decode()).encode(),
        "runtime/supervisor.py": instrument_supervisor(members["runtime/supervisor.py"].decode()).encode(),
        "runtime/fd_permission.py": (HERE / "fd_permission.py").read_bytes(),
    }
    for name, code in new.items():
        compile(code, name, "exec")
    return new
