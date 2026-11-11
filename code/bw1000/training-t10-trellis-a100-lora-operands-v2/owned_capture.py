"""Identify and stop only the exact native reference command and its descendants."""
import os,signal,time
from pathlib import Path
def snapshot():
    rows={}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            v=(p/"stat").read_text().split(") ",1)[1].split()
            rows[int(p.name)]={"PID":int(p.name),"PPID":int(v[1]),"PGID":int(v[2]),"startticks":int(v[19]),
                              "state":v[0],"args":(p/"cmdline").read_bytes().split(b"\0")}
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return rows
def native_command(args,root):
    capture=str(root/'src/capture_a100_lora.py').encode()
    bounded=str(root/'src/bounded_capture.py').encode()
    return (capture in args or bounded in args) and str(root/'capture').encode() in args

def refresh(registry,root):
    rows=snapshot()
    owned={pid for pid,saved in registry.items() if pid in rows and rows[pid]["startticks"]==saved["startticks"]}
    owned.update(pid for pid,v in rows.items() if native_command(v["args"],root))
    changed=True
    while changed:
        fresh={pid for pid,v in rows.items() if v["PPID"] in owned}
        changed=not fresh.issubset(owned);owned.update(fresh)
    for pid in owned:
        v=rows[pid]
        registry[pid]={k:v[k] for k in ("PID","PPID","PGID","startticks","state")}
    return {pid:rows[pid] for pid in owned if rows[pid]["state"]!="Z"}
def signal_owned(registry,root,kind):
    for pid,v in refresh(registry,root).items():
        if pid==os.getpid():raise RuntimeError("supervisor cannot signal itself")
        try:
            current=snapshot().get(pid)
            if current and current["startticks"]==v["startticks"]:os.kill(pid,kind)
        except ProcessLookupError:pass
def cleanup(registry,root,grace=10):
    signaled=bool(refresh(registry,root))
    if signaled:signal_owned(registry,root,signal.SIGTERM)
    until=time.monotonic()+grace
    while refresh(registry,root) and time.monotonic()<until:time.sleep(.25)
    if refresh(registry,root):signal_owned(registry,root,signal.SIGKILL)
    until=time.monotonic()+grace
    while refresh(registry,root) and time.monotonic()<until:time.sleep(.25)
    return {"owned_processes_remaining":len(refresh(registry,root)),"termination_needed":signaled}
