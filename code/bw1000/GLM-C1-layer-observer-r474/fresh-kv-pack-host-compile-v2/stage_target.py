"""Send exactly one complete v2 source bundle through the existing bounded executor."""

import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


CLAIM_MEMBER = "code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-host-compile-v2/HOST_CLAIM.json"
CLAIM_SHA256 = "95e7cb7accf7cdc14ab028d58f2d78b460dd6a06a2d1e541439380193f3bafb4"
REQUEST = "d0b051b52122d2858ca72058d217d915"
LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v2"
MAX_PROGRAM = 131072
MAX_JSON = 220000

RECEIVER = r'''
import base64
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import zlib


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def receive(root, archive):
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(base64.b64decode(archive, validate=True), MAX_JSON + 1)
    require(len(raw) <= MAX_JSON and decompressor.eof and not decompressor.unused_data,
            "reviewed source archive exceeds its bound")
    encoded = json.loads(raw)
    require(isinstance(encoded, dict) and len(encoded) == 23, "complete source inventory required")
    files = {}
    for name, value in encoded.items():
        path = PurePosixPath(name)
        require(path.as_posix() == name and not path.is_absolute() and ".." not in path.parts
                and path.parts[0] in ("code", "coordination", "tools"), "invalid source member")
        files[path] = base64.b64decode(value, validate=True)
    member = PurePosixPath(CLAIM_MEMBER)
    claim = files[member]
    require(sha256(claim).hexdigest() == CLAIM_SHA256, "published v2 host claim changed")
    scope = json.loads(claim)
    require(scope["request"] == REQUEST and scope["label"] == LABEL
            and scope["files_SHA256"] == {name.as_posix(): sha256(data).hexdigest()
                                         for name, data in files.items() if name != member},
            "reviewed package references changed or missing bytes")
    target = root / "incoming" / (LABEL + "-reviewed-checkout")
    require(target.parent.resolve(strict=True) == target.parent, "unexpected source staging root")
    if target.exists() or target.is_symlink():
        require(target.resolve(strict=True) == target and target.is_dir(), "invalid existing v2 source")
        observed = set(target.rglob("*"))
        require({name.relative_to(target).as_posix() for name in observed if name.is_file()} ==
                {name.as_posix() for name in files} and not any(name.is_symlink() for name in observed)
                and all((target / str(name)).read_bytes() == data for name, data in files.items()),
                "existing v2 source scope is incomplete or differs")
        state = "V2_SOURCE_ALREADY_IDENTICAL"
    else:
        with TemporaryDirectory(dir=target.parent, prefix="glm-fresh-kv-v2-") as temporary:
            stage = Path(temporary) / "checked"
            stage.mkdir(mode=0o700)
            for name, data in files.items():
                path = stage.joinpath(*name.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(data)
            require(not target.exists() and not target.is_symlink(), "v2 stage appeared during staging")
            stage.rename(target)
        state = "V2_SOURCE_STAGED"
    return dict(request=REQUEST, label=LABEL, state=state, claim_SHA256=CLAIM_SHA256,
                stage_manifest_SHA256=scope["stage_manifest_SHA256"], staged_files=len(files), actual_new_Slurm_steps=0)


if __name__ == "__main__":
    project = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
    print(json.dumps(receive(project, ARCHIVE), allow_nan=False), flush=True)
'''


def program(reviewed):
    reviewed = Path(reviewed).resolve(strict=True)
    claim = (reviewed / CLAIM_MEMBER).read_bytes()
    if sha256(claim).hexdigest() != CLAIM_SHA256:
        raise ValueError("reviewed v2 claim differs")
    pins = json.loads(claim)["files_SHA256"]
    content = {CLAIM_MEMBER: claim}
    for name, pin in pins.items():
        path = reviewed / name
        if path.resolve(strict=True) != path:
            raise ValueError("noncanonical staged source path")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != pin:
            raise ValueError(f"reviewed v2 member differs: {name}")
        content[name] = raw
    raw_json = json.dumps({name: base64.b64encode(raw).decode("ascii") for name, raw in content.items()},
                          sort_keys=True, separators=(",", ":")).encode("ascii")
    if len(raw_json) > MAX_JSON:
        raise ValueError("reviewed source inventory exceeds the compressed envelope")
    archive = base64.b64encode(zlib.compress(raw_json, 9)).decode("ascii")
    constants = dict(ARCHIVE=archive, CLAIM_SHA256=CLAIM_SHA256, CLAIM_MEMBER=CLAIM_MEMBER,
                     REQUEST=REQUEST, LABEL=LABEL, MAX_JSON=MAX_JSON)
    body = ("\n".join(f"{key}={value!r}" for key, value in constants.items()) + "\n" + RECEIVER).encode("ascii")
    if len(body) > MAX_PROGRAM:
        raise ValueError("reviewed source exceeds the existing one-program transport bound")
    compile(body, "<verified-v2-stage-packet>", "exec")
    return body
