"""Build the reviewed OFF train-only bundle and derive the native CPU entry."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import stat
import time
from off_inputs import check_row

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "training-minicpm-usgs-native88-CPU-v1"
TRAIN_SHA = "90b39323ebd444159c4a4f880ecc98fbaccaae662d1e3d6d95c4e69c5d7ab31a"
REVIEW_SHA = "c3a50c990597fef7b2893c7eb96a42b98d9176bba66b33ce9076879eabfe0e45"
MEDIA = [
    (526159, "3469db446c30861b1b3d01d7b1e411bf39413c740d853230a8b1f58095a6a9bb"),
    (538705, "7beaeb860ce80f09dbe0e98b1371960fa2c1b23b60d8946a1beaeed614af3dea"),
    (278867, "7ac4d2376af38f3094b9ddbd35ed4bcbeac061b4c296e31d27713a63d63dddca"),
    (1144830, "a00c4a644329302dbc3182033e406778a4fe3ce6b7cdf1ecac16848d780befa1"),
]
PINS = {
    "ASSETS.json": "16bd66c3df682fcaa3ea6cc0f66739c968b819a37d6511e97a50cd7ab03f83e7",
    "controller.py": "cbad79017df4de66e408fad1f9ff4e534da1ec3d14e922037528dd59e486d640",
    "input_contract.py": "f0202aaa5f1ff3627de9d95c9f2d149584afa3f14a86fe75ed5e6c6799923edf",
    "providers.py": "4196a19e4f3fae4eb399ef76f752bbd4f009267abaebd4379237d71373951479",
    "slurm_accounting.py": "15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b",
    "worker.py": "543ecc750f6a4dcee80a0cc6001bfd613d2bfdfe5fd2ed91d638531cd10409ca",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha(raw)}


def require(test, code):
    if not test:
        raise ValueError(code)


def read_pinned(path, expected, size=None):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and (size is None or before.st_size == size), "INPUT_FILE_TYPE_OR_SIZE")
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after) and sha(raw) == expected, "INPUT_FILE_IDENTITY")
    return raw


def train_files(origin):
    origin = origin.resolve(strict=True)
    reviewed = read_pinned(origin / "reviewed_manifest.json", REVIEW_SHA, 1081)
    raw = read_pinned(origin / "train.swift.jsonl", TRAIN_SHA, 2617)
    lines = raw.splitlines()
    require(len(lines) == len(set(lines)) == 6, "TRAIN_ROWS")
    files, rows, media, provenance = {}, [], {}, []
    for i, line in enumerate(lines, 1):
        row = json.loads(line)
        require(set(row) == {"images", "messages"} and len(row["images"]) == 1, "TRAIN_SCHEMA")
        source = Path(row["images"][0])
        require(source.is_absolute() and source.parent == origin and re.fullmatch(r"[0-9]+-[0-9]+\.jpg", source.name), "TRAIN_SOURCE_IMAGE")
        if source not in media:
            require(len(media) < len(MEDIA), "TRAIN_MEDIA_COUNT")
            name = f"off-image-{len(media) + 1:02d}.jpg"
            size, digest = MEDIA[len(media)]
            files[name] = read_pinned(source, digest, size)
            media[source] = name
        converted = dict(id=f"off-train-{i:02d}", images=[media[source]], messages=row["messages"])
        classification = check_row(converted, i)
        # Each decoded UTF-8 message is retained verbatim. Hash original JSONL rows separately.
        require(json.loads(json_bytes(converted))["messages"] == row["messages"], "TRAIN_REWRITE")
        rows.append(json_bytes(converted))
        provenance.append({"ordinal": i, "original_row_SHA256": sha(line), "staged_row_SHA256": sha(rows[-1].rstrip(b"\n")),
                           "image": media[source], "answer_class": classification})
    require(len(media) == 4 and len({sha(x) for x in files.values()}) == 4
            and sum(x["answer_class"] == "numeric_g" for x in provenance) == 4, "TRAIN_COUNTS")
    files["train.jsonl"] = b"".join(rows)
    manifest = dict(role="OFF_TRAIN_ONLY", rows=provenance, planned=6, unique_media=4,
                    source_train_SHA256=sha(raw), reviewed_manifest_SHA256=sha(reviewed),
                    files={name: pin(body) for name, body in files.items()})
    files["INPUT_FILES.json"] = json_bytes(manifest)
    return files


def change(source, before, after):
    require(source.count(before) == 1, "ORIGINAL_PATCH_BOUNDARY")
    return source.replace(before, after, 1)


def render(origin, request, parent, node, publication=None, authorization=None):
    require(re.fullmatch(r"[0-9a-f]{32}", request) and request != "7f9a0e2809bf4f53a7294fdc3dc74225", "NEW_REQUEST")
    require(re.fullmatch(r"[1-9][0-9]+", parent) and re.fullmatch(r"f[0-9]+r[0-9]+n[0-9]+", node), "PLACEMENT_IDENTITY")
    require(publication is None or re.fullmatch(r"[0-9a-f]{40}", publication), "PUBLICATION_IDENTITY")
    require((publication is None) == (authorization is None), "PUBLICATION_AND_AUTHORIZATION_REQUIRED")
    train = train_files(origin)
    files = {name: read_pinned(ORIGINAL / name, digest) for name, digest in PINS.items()}
    label = f"training-minicpm-off-native6-cpu-{request[:12]}"
    data_sha = sha(train["INPUT_FILES.json"])
    if authorization is not None:
        require(set(authorization) == {"request", "parent", "node", "publication", "input_manifest_SHA256",
                "CPU_GO", "GPU_GO", "training_GO", "Model_GO", "data_on_target_authorized"}
                and authorization == dict(request=request, parent=parent, node=node, publication=publication,
                input_manifest_SHA256=data_sha, CPU_GO=True, GPU_GO=False, training_GO=False, Model_GO=False,
                data_on_target_authorized=True), "EXACT_ROOT_AUTHORIZATION")
    scope = dict(request=request, label=label, parent=parent, node=node, publication=publication,
                 input_manifest_SHA256=data_sha, CPU_GO=publication is not None, root_GO=publication is not None,
                 GPU_GO=False, training_GO=False, Model_GO=False,
                 authorization_SHA256=sha(json_bytes(authorization)) if authorization is not None else None)
    assets = json.loads(files["ASSETS.json"])
    del assets["media"], assets["dataset_SHA256"]
    files["ASSETS.json"] = json_bytes(assets)
    files["CLAIM.json"] = json_bytes(scope)
    files["off_inputs.py"] = (HERE / "off_inputs.py").read_bytes()
    worker = files["worker.py"].decode()
    worker = change(worker, "from input_contract import check_record,check_encoded",
                    "from input_contract import check_encoded\nfrom off_inputs import read_inputs")
    start = worker.index("DATA=Path(")
    finish = worker.index("VERSIONS=", start)
    worker = worker[:start] + 'DATA=Path("/off-train")\nMODEL=ROOT/"metadata-only/MiniCPM-V-4.6"\n' + f'DATA_SHA256="{data_sha}"\n' + worker[finish:]
    start = worker.index("        raw=DATA.read_bytes()")
    finish = worker.index('        phase="native-provider"', start)
    worker = worker[:start] + '        rows=read_inputs(DATA,DATA_SHA256)\n' + worker[finish:]
    worker = change(worker, '    deadline=min(intent["deadline"]-55,time.time()+210)',
                    f'    if intent["request"]!="{request}" or intent["input_manifest_SHA256"]!=DATA_SHA256:\n'
                    '        raise ValueError("OFF_WORKER_SCOPE")\n'
                    '    deadline=min(intent["deadline"]-55,time.time()+210)')
    worker = change(worker, '            check_record(row)\n', '')
    worker = re.sub(r"\b(88|176|22)\b", lambda m: {"88": "6", "176": "12", "22": "4"}[m[0]], worker)
    worker = worker.replace("MINICPM_USGS88_NATIVE_INPUT_CPU_COMPLETE", "MINICPM_OFF6_NATIVE_INPUT_CPU_COMPLETE")
    worker = worker.replace('message=str(exc)[:2048],', '')
    files["worker.py"] = worker.encode()
    controller = files["controller.py"].decode()
    controller = change(controller, 'LABEL="training-minicpm-usgs-native88-CPU-v1"', f'LABEL="{label}"')
    controller = change(controller, 'REQUEST="7f9a0e2809bf4f53a7294fdc3dc74225"', f'REQUEST="{request}"')
    controller = change(controller, 'PARENT,NODE="1103606","f13r1n19"', f'PARENT,NODE="{parent}","{node}"')
    controller = change(controller, '    RUN.mkdir(mode=0o700)', f'''    need(claim["publication"]==payload["publication"] and claim["parent"]==PARENT and claim["node"]==NODE
         and claim["input_manifest_SHA256"]=="{data_sha}" and claim["authorization_SHA256"]=={scope["authorization_SHA256"]!r}
         and claim["authorization_SHA256"] is not None and time.time()-epoch<=45,"OFF_OWNER_SCOPE")
    probe=R/"incoming"/LABEL/"train"
    need(probe.resolve(strict=True)==probe and probe.is_dir(),"OFF_STAGE_SCOPE")
    RUN.mkdir(mode=0o700)''')
    controller = change(controller, 'parent=PARENT,node=NODE,planned_rows=', f'parent=PARENT,node=NODE,input_manifest_SHA256="{data_sha}",planned_rows=')
    start = controller.index('    data=R/"results/training-domain-usgs-table-reading-v1/data/train"')
    finish = controller.index("    framework=", start)
    controller = controller[:start] + f'''    from off_inputs import read_inputs
    data=probe
    read_inputs(data,"{data_sha}")
''' + controller[finish:]
    controller = change(controller, '(data,data,"ro")', '(data,"/off-train","ro")')
    controller = change(controller, 'new_assets_transferred=False', 'train_input_staged_separately=True')
    controller = change(controller, 'row[2:4]==["COMPLETED","0:0"]', 'row[0].startswith(PARENT+".") and row[0].rsplit(".",1)[1].isdecimal() and row[2:4]==["COMPLETED","0:0"]')
    controller = re.sub(r"\b(88|176|22)\b", lambda m: {"88": "6", "176": "12", "22": "4"}[m[0]], controller)
    controller = change(controller, '    save("step-intent.json",dict(command=command',
        '''    need(accounting() is None,"OFF_STEP_LABEL_CONFLICT")
    for line in query("squeue","--steps","-h","-j",PARENT,"-o","%i|%N").splitlines():
        step,nodes=line.split("|",1)
        if step in (PARENT+".batch",PARENT+".extern"):
            continue
        need(step.startswith(PARENT+".") and step.rsplit(".",1)[1].isdecimal()
             and nodes and NODE not in query("scontrol","show","hostnames",nodes).splitlines(),"OFF_STEP_QUEUE_CONFLICT")
    for line in query("scontrol","show","step",PARENT,"-o").splitlines():
        fields=dict(x.split("=",1) for x in line.split() if "=" in x)
        step=fields["StepId"]
        if step in (PARENT+".batch",PARENT+".extern"):
            continue
        need(step.startswith(PARENT+".") and step.rsplit(".",1)[1].isdecimal()
             and NODE not in query("scontrol","show","hostnames",fields["NodeList"]).splitlines(),"OFF_STEP_CONTROLLER_CONFLICT")
    need(time.time()-epoch<=45,"OFF_FRESHNESS_EXPIRED_BEFORE_SRUN")
    save("step-intent.json",dict(command=command''')
    files["controller.py"] = controller.encode()
    files["independent.py"] = (HERE / "independent.py").read_bytes()
    files["SOURCE_FILES.json"] = json_bytes({"files": {name: pin(body) for name, body in files.items()}})
    for name, body in files.items():
        if name.endswith(".py"):
            compile(body, name, "exec")
    return files, train, scope


def main():
    parser = argparse.ArgumentParser()
    for option in ("source", "output"):
        parser.add_argument("--" + option, type=Path)
    for option in ("request", "parent", "node"):
        parser.add_argument("--" + option)
    parser.add_argument("--publication")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--controller-payload", type=Path,
                        help="Write the controller bytes as a separate file using the current UTC epoch")
    parser.add_argument("--payload-from", type=Path,
                        help="Render the controller from an existing, independently verified src directory")
    parser.add_argument("--source-manifest")
    args = parser.parse_args()
    if args.payload_from:
        if args.controller_payload is None or not args.source_manifest:
            parser.error("--payload-from requires --controller-payload and --source-manifest")
        require(sha((args.payload_from / "SOURCE_FILES.json").read_bytes()) == args.source_manifest,
                "OFF_PREPARED_SOURCE_MANIFEST_SHA")
        payload = make_payload(args.payload_from, time.time())
        with args.controller_payload.open("xb") as output:
            output.write(payload)
        print(json.dumps({"controller_bytes": pin(payload)}, sort_keys=True))
        return
    if not all((args.source, args.output, args.request, args.parent, args.node)):
        parser.error("--source, --output, --request, --parent and --node are required for generation")
    authorization = json.loads(args.authorization.read_bytes()) if args.authorization else None
    source, train, scope = render(args.source, args.request, args.parent, args.node, args.publication, authorization)
    args.output.mkdir(mode=0o700)
    for directory, members in (("src", source), ("train", train)):
        target = args.output / directory
        target.mkdir(mode=0o700)
        for name, raw in members.items():
            with (target / name).open("xb") as output:
                output.write(raw)
    if args.controller_payload:
        with args.controller_payload.open("xb") as output:
            output.write(make_payload(args.output / "src", time.time()))
    print(json.dumps(dict(request=scope["request"], label=scope["label"], source_manifest=pin(source["SOURCE_FILES.json"]),
                          input_manifest=pin(train["INPUT_FILES.json"]), input_member_count=len(train),
                          input_bytes=sum(map(len, train.values())), disabled=not scope["CPU_GO"]), sort_keys=True))


def make_payload(src, epoch):
    manifest = (src / "SOURCE_FILES.json").read_bytes()
    pins = json.loads(manifest)["files"]
    files = {name: read_pinned(src / name, spec["SHA256"], spec["bytes"]) for name, spec in pins.items()}
    files["SOURCE_FILES.json"] = manifest
    claim = json.loads(files["CLAIM.json"])
    require(claim["CPU_GO"] is True and claim["authorization_SHA256"] is not None, "OFF_UNADOPTED")
    payload = dict(files={name: base64.b64encode(raw).decode("ascii") for name, raw in files.items()},
                   request=claim["request"], publication=claim["publication"], epoch=epoch, deadline=epoch+300,
                   source_manifest_SHA256=sha(manifest))
    # The original controller expects a single injected stdlib payload.
    return b"PAYLOAD=" + repr(payload).encode("ascii") + b"\n" + files["controller.py"]

if __name__ == "__main__":
    main()
