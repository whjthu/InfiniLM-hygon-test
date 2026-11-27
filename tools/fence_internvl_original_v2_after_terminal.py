"""Exclusively close the original unresolved InternVL request after standby C0."""
import argparse
import json
from pathlib import Path
import subprocess
from types import ModuleType
from uuid import uuid4

import fence_internvl_original_v2 as original
from run_hygon_bound_payload import digest, document, send


REQUEST = "6063ab4872014f7dbbee8db7bd97abaa"
STANDBY_STOP_UTC = "2026-11-24T10:14:48"


def replace(program, old, new):
    if program.count(old) != 1:
        raise ValueError("Unexpected frozen original fence program")
    return program.replace(old, new)


def render():
    program = replace(original.HOST, original.REQUEST, REQUEST)
    program = replace(program, "packet=json.loads(sys.stdin.buffer.read(4096))",
                      "os.environ.update(TZ='UTC',LC_ALL='C')\npacket=json.loads(sys.stdin.buffer.read(4096))")
    previous_live = """    assert len(live)==1 and [live[0][k] for k in ('StepId','Name','NodeList','State','TRES')]==[
        standby,'training80-dcu-standby-b71dc03b40ad-f13r1n14',node,'RUNNING','cpu=8,gres/dcu=8,mem=16G,node=1']"""
    program = replace(program, previous_live, "    assert not live, 'standby or original workload is still live'")
    program = replace(program, "2026-11-24T13:00:00", "2026-11-24T00:00:00")
    program = replace(program,
                      "    assert old_standby[2:4]==['RUNNING','0:0'] and old_standby[5:7]==[node,'cpu=8,gres/dcu=8,mem=16G,node=1']",
                      "    assert old_standby[2:5]==['COMPLETED','0:0','10820'] and "
                      "old_standby[5:9]==[node,'cpu=8,gres/dcu=8,mem=16G,node=1',"
                      "'2026-11-24T07:14:28'," + repr(STANDBY_STOP_UTC) + "]")
    compile(program, "<original InternVL terminal fence>", "exec")
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = parser.parse_args()
    source_sha = original.exact_source()
    program = render()
    if args.check:
        print(json.dumps({"request": REQUEST, "original_launch_sha256": source_sha,
                          "fence_source_sha256": digest(program.encode()), "remote_calls": 0}))
        return 0
    repo = original.G
    for path in (Path(__file__).resolve(), Path(original.__file__).resolve(), repo / "tools/run_hygon_bound_payload.py"):
        published = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "show",
             args.publication + ":" + path.relative_to(repo).as_posix()], cwd=repo,
        )
        if published != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Publication differs from the exact local code")
    local = repo / "local-only"
    connector = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = connector.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("Existing authenticated SSH helper differs")
    transport = ModuleType("existing_ssh")
    exec(compile(raw, str(connector), "exec"), vars(transport))
    request = {"request": REQUEST, "original": original.ORIGINAL,
               "original_publication": original.ORIGINAL_PUBLICATION, "original_launch_sha256": source_sha}
    result = send(transport.ssh(repo.parent), request=REQUEST, program=program, data=document(request),
                  send_token=local / f"internvl-terminal-fence-{REQUEST}-send.json", outcome_seconds=45)
    path = local / f"internvl-terminal-fence-{REQUEST}-{uuid4().hex}.json"
    data = document(result)
    with path.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"file": path.name, "sha256": digest(data), **result}, sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED")


if __name__ == "__main__":
    raise SystemExit(main())
