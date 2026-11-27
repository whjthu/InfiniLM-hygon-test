"""Stage the sealed CP25 public evaluation inputs and verify the native CPU template."""

import argparse
from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from types import ModuleType
from zipfile import ZIP_STORED, ZipFile

import launch_internvl_gqa_val96_cpu_once as cpu
from run_hygon_bound_payload import digest, document, send
import stage_internvl_gqa_val96_once as upload


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LOCAL = REPO / "local-only"
STAGE_LABEL = "training-internvl-gqa56-cp25-confirmatory-val96-user-only-v1"
CPU_LABEL = "training-internvl-gqa56-cp25-confirmatory-val96-native-cpu-v1"
STAGE_REQUEST = "05cef59bc7a04032bb0257826731db9a"
CPU_REQUEST = "66dd35da70a149358183283f13b98847"
INPUT_SHA = "b28894645f035dda3ccaeb27e6cfa606c7dfc3bbcbaf657d4c79d0f5f93eba38"
PROMPTS_SHA = "8ff40c84f5eaf2995234b834bd2c37353b6a6ec99d155f38af55738a15d9dc8f"
PACKET_SHA = "0f9811b120af4a82db92438d8a1dad7f4c79731e2aab1488ba18a17deeca8e36"
TRANSPORT_SHA = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
PACKET_BYTES = 15_044_033
INPUT_BYTES = 15_033_263


def replace_one(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Expected exactly one source rebind site")
    return source.replace(before, after, 1)


def public_packet():
    path = LOCAL / "internvl-gqa56-cp25-confirmatory-val96-stage-v1/input.zip"
    raw = upload.evaluation.read(path, PACKET_BYTES, PACKET_SHA)
    with ZipFile(BytesIO(raw)) as archive:
        index = archive.read("INPUT.json")
        if digest(index) != INPUT_SHA:
            raise ValueError("Frozen input index differs")
        manifest = json.loads(index)
        prompts = archive.read("prompts.json")
        names = {"INPUT.json", "prompts.json", *(row["image"] for row in upload.evaluation.prompt_rows(prompts))}
        records = archive.infolist()
        if (set(archive.namelist()) != names or len(records) != len(names) or len(names) != 98
                or set(manifest["members"]) != names - {"INPUT.json"}
                or manifest["gold_included"] is not False or digest(prompts) != PROMPTS_SHA
                or any(record.compress_type != ZIP_STORED for record in records)
                or sum(record.file_size for record in records) != INPUT_BYTES):
            raise ValueError("Public-only payload differs")
        for name, pin in manifest["members"].items():
            if upload.evaluation.fields(archive.read(name)) != pin:
                raise ValueError("Public-only input member differs")
    return raw


def bindings(values, template):
    code = "\n".join(f"{name}={value!r}" for name, value in values.items()) + "\n" + template
    compile(code, "<CP25 confirmatory HOST>", "exec")
    return code


def stage_program(publication):
    return bindings(dict(REQUEST=STAGE_REQUEST, LABEL=STAGE_LABEL, INPUT_SHA=INPUT_SHA,
                         INPUT_BYTES=INPUT_BYTES, PACKET_BYTES=PACKET_BYTES, PACKET_SHA=PACKET_SHA,
                         PUBLICATION=publication), upload.REMOTE)


def cpu_package(publication, upload_pin):
    sources = {}
    for name in cpu.MEMBERS:
        src = (cpu.SOURCE / name).read_text(encoding="utf-8")
        if name == "internvl_gqa_val96.py":
            src = replace_one(src, f'PROMPTS_SHA = "{upload.evaluation.PROMPTS_SHA}"',
                              f'PROMPTS_SHA = "{PROMPTS_SHA}"')
        elif name == "internvl_gqa_val96_native.py":
            src = replace_one(src, f'INPUT_SHA = "{cpu.INPUT_PIN}"', f'INPUT_SHA = "{INPUT_SHA}"')
        compile(src, name, "exec")
        sources[name] = src.encode()
    template = replace_one(cpu.REMOTE, 'book = results / "training-internvl-gqa-val96-user-only-v1"',
                           f'book = results / "{STAGE_LABEL}"')
    program = bindings(dict(REQUEST=CPU_REQUEST, LABEL=CPU_LABEL, MEMBERS=cpu.MEMBERS,
                            UPLOAD_PIN=upload_pin, INPUT_PIN=INPUT_SHA), template)
    payload = document(dict(publication=publication, files={name: b64encode(raw).decode()
                                                            for name, raw in sources.items()}))
    return program, payload


def verify_publication(publication):
    if not re.fullmatch("[0-9a-f]{40}", publication):
        raise ValueError("Exact published source revision required")
    paths = (Path(__file__).resolve(), HERE / "run_hygon_bound_payload.py", Path(cpu.__file__),
             Path(upload.__file__), *(cpu.SOURCE / name for name in cpu.MEMBERS))
    for path in paths:
        key = path.relative_to(REPO).as_posix()
        raw = subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}",
                                       "show", f"{publication}:{key}"], cwd=REPO)
        if raw != path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Published bytes differ for {key}")


def stage_pin(path, publication):
    result = json.loads(upload.evaluation.read(path, 1 << 17))
    response = result.get("response") or {}
    expected = dict(request=STAGE_REQUEST, rc=0, input_SHA256=INPUT_SHA,
                    archive_SHA256=PACKET_SHA, images=96, files=98, published_source=publication,
                    gold_included=False, files_read_back=98, compute_native_executed=False)
    if (result["remote_outcome"] != "RESPONSE_OBSERVED" or result["remote_received"] is not True
            or set(response) != {*expected, "receipt_SHA256"}
            or any(response[name] != value for name, value in expected.items())):
        raise ValueError("A successful original upload receipt is required")
    pin = digest((json.dumps(expected, sort_keys=True) + "\n").encode())
    if response["receipt_SHA256"] != pin:
        raise ValueError("Original upload receipt hash differs")
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "stage", "cpu"))
    parser.add_argument("--publication")
    args = parser.parse_args()
    if args.mode == "check":
        raw = public_packet()
        host, payload = cpu_package("0" * 40, "0" * 64)
        print(json.dumps(dict(archive_SHA256=digest(raw), bytes=len(raw), images=96,
                              stage_host_SHA256=digest(stage_program("0" * 40).encode()),
                              cpu_host_SHA256=digest(host.encode()), cpu_payload_SHA256=digest(payload),
                              network=False), sort_keys=True))
        return 0
    verify_publication(args.publication or "")
    stage_outcome = LOCAL / f"internvl-cp25-confirmatory-stage-{STAGE_REQUEST}-result.json"
    if args.mode == "stage":
        request, result_path = STAGE_REQUEST, stage_outcome
        source, data = stage_program(args.publication), public_packet()
        limits = dict(packet_cap=30 << 20, write_seconds=180, outcome_seconds=75)
    else:
        request = CPU_REQUEST
        result_path = LOCAL / f"internvl-cp25-confirmatory-cpu-{request}-result.json"
        source, data = cpu_package(args.publication, stage_pin(stage_outcome, args.publication))
        limits = dict(outcome_seconds=224)
    if result_path.exists():
        raise ValueError("Original outcome is already recorded")
    helper = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = upload.evaluation.read(helper, 1 << 18, TRANSPORT_SHA)
    transport = ModuleType("existing_ssh")
    exec(compile(raw, str(helper), "exec"), vars(transport))
    result = send(transport.ssh(REPO.parent), request=request, program=source, data=data,
                  send_token=LOCAL / f"internvl-cp25-confirmatory-{request}-send.json", **limits)
    upload.evaluation.write(result_path, document(result))
    response = result.get("response") or {}
    print(json.dumps(dict(request=request, receipt=result_path.name, remote_outcome=result["remote_outcome"],
                          remote_received=result["remote_received"], response=response), sort_keys=True))
    return int(result["remote_outcome"] != "RESPONSE_OBSERVED" or response.get("rc") != 0)


if __name__ == "__main__":
    raise SystemExit(main())
