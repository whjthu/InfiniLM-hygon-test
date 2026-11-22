"""Package the large, frozen Native checkout for the existing bounded executor."""

import base64
from hashlib import sha256
import json
from pathlib import Path
import zlib


LABEL = "GLM474-fresh-kv-explicit-ar-Native-CPU-v2"
REQUEST = "0da040e34559841c9f94ebd520a63944"
CLAIM_SHA = "f8c11a6780821c76c59f79ed17f01cd06640f1422aa39cc7ef4796bc7d0777a9"
MANIFEST_SHA = "ed14552b184a3e6e29bb0315a9f337e4030228bf17fb217119f9ebbe0893196b"
SCOPE = "code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-explicit-ar-native-host-v2"
MANIFEST_MEMBER = SCOPE + "/STAGE_FILES.json"
CLAIM_MEMBER = SCOPE + "/SCOPE.json"
CHUNK_MAX = 90000
WIRE_MAX = 131072
MAX_JSON = 3 * 1024**2

RECEIVER = r'''
import base64
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import zlib


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def regular(path, limit):
    need(path.resolve(strict=True) == path and path.is_file(), "transfer requires original canonical files")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit, "transfer bytes exceed their bound")
    return raw


def receive(root):
    incoming = root / "incoming"
    need(incoming.resolve(strict=True) == incoming and incoming.is_dir(), "transfer parent differs")
    prefix = incoming / (LABEL + "-transfer-" + MANIFEST_SHA[:12])
    target = incoming / (LABEL + "-reviewed-checkout")
    need(1 <= len(CHUNKS) <= 12, "transfer inventory exceeds its bound")
    def chunk_path(index):
        return prefix / f"chunk-{index:02}.bin"
    def metadata(state, **values):
        return dict(state=state, request=REQUEST, stage_manifest_SHA256=MANIFEST_SHA,
                    actual_new_Slurm_steps=0, **values)
    if INDEX is not None:
        need(type(INDEX) is int and 0 <= INDEX < len(CHUNKS), "unexpected transfer index")
        raw = base64.b64decode(PAYLOAD, validate=True)
        need(0 < len(raw) <= CHUNK_MAX and digest(raw) == CHUNKS[INDEX], "transfer chunk identity differs")
        prefix.mkdir(mode=0o700, exist_ok=True)
        need(prefix.resolve(strict=True) == prefix and prefix.is_dir(), "invalid transfer scope")
        path = chunk_path(INDEX)
        if path.exists() or path.is_symlink():
            need(regular(path, CHUNK_MAX) == raw, "existing transfer chunk differs")
            return metadata("V2_SOURCE_CHUNK_IDENTICAL", index=INDEX, chunk_SHA256=CHUNKS[INDEX])
        with path.open("xb") as stream:
            stream.write(raw)
        return metadata("V2_SOURCE_CHUNK_STAGED", index=INDEX, chunk_SHA256=CHUNKS[INDEX])
    need(prefix.resolve(strict=True) == prefix and prefix.is_dir(), "original transfer scope absent")
    parts = [regular(chunk_path(index), CHUNK_MAX) for index in range(len(CHUNKS))]
    need([digest(chunk) for chunk in parts] == CHUNKS, "original transfer chunk bytes changed")
    archive = b"".join(parts)
    need(digest(archive) == ARCHIVE_SHA, "complete source archive changed")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(archive, MAX_JSON + 1)
    need(len(raw) <= MAX_JSON and decoder.eof and not decoder.unused_data, "source archive output exceeds bound")
    encoded = json.loads(raw)
    need(isinstance(encoded, dict) and len(encoded) == 33, "source archive file count differs")
    files = {}
    for name, encoded_bytes in encoded.items():
        path = PurePosixPath(name)
        need(path.as_posix() == name and not path.is_absolute() and ".." not in path.parts
             and path.parts and path.parts[0] in ("code", "coordination", "tools", "results"),
             "invalid original source member")
        files[name] = base64.b64decode(encoded_bytes, validate=True)
    manifest = files[MANIFEST_MEMBER]
    need(digest(manifest) == MANIFEST_SHA and digest(files[CLAIM_MEMBER]) == CLAIM_SHA
         and json.loads(manifest)["files_SHA256"] ==
         {name: digest(data) for name, data in files.items() if name != MANIFEST_MEMBER},
         "complete reviewed bytes do not match the published scope")
    if target.exists() or target.is_symlink():
        need(target.resolve(strict=True) == target and target.is_dir(), "invalid reviewed scope")
        tree = list(target.rglob("*"))
        need(not any(path.is_symlink() for path in tree)
             and {path.relative_to(target).as_posix() for path in tree if path.is_file()} == files.keys()
             and all(regular(target / name, len(data)) == data for name, data in files.items()),
             "existing reviewed checkout differs")
        return metadata("V2_SOURCE_ALREADY_IDENTICAL", files=len(files), archive_SHA256=ARCHIVE_SHA)
    with TemporaryDirectory(dir=incoming, prefix="glm-explicit-ar-v2-") as temp:
        stage = Path(temp) / "verified"
        stage.mkdir(mode=0o700)
        for name, data in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        need(not target.exists() and not target.is_symlink(), "reviewed checkout appeared")
        stage.rename(target)
    return metadata("V2_SOURCE_STAGED", files=len(files), archive_SHA256=ARCHIVE_SHA)


if __name__ == "__main__":
    root = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
    print(json.dumps(receive(root), allow_nan=False), flush=True)
'''


def digest(raw):
    return sha256(raw).hexdigest()


def programs(checkout):
    """Return ordered source-chunk programs plus the single final installation program."""
    source = Path(checkout).resolve(strict=True)
    manifest = (source / MANIFEST_MEMBER).read_bytes()
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError("v2 publication stage manifest differs")
    content = {MANIFEST_MEMBER: manifest}
    for name, pin in json.loads(manifest)["files_SHA256"].items():
        path = source / name
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("reviewed staging path differs")
        raw = path.read_bytes()
        if digest(raw) != pin:
            raise ValueError("reviewed staging member differs")
        content[name] = raw
    bundle = json.dumps({name: base64.b64encode(data).decode("ascii") for name, data in content.items()},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    if digest(content[CLAIM_MEMBER]) != CLAIM_SHA or len(content) != 33 or len(bundle) > MAX_JSON:
        raise ValueError("reviewed source inventory or claim differs")
    archive = zlib.compress(bundle, 9)
    chunks = [archive[start:start+CHUNK_MAX] for start in range(0, len(archive), CHUNK_MAX)]
    meta = dict(LABEL=LABEL, REQUEST=REQUEST, CLAIM_SHA=CLAIM_SHA, MANIFEST_SHA=MANIFEST_SHA,
                CLAIM_MEMBER=CLAIM_MEMBER, MANIFEST_MEMBER=MANIFEST_MEMBER, ARCHIVE_SHA=digest(archive),
                CHUNKS=[digest(chunk) for chunk in chunks], CHUNK_MAX=CHUNK_MAX, MAX_JSON=MAX_JSON)
    result = []
    for index in [*range(len(chunks)), None]:
        values = dict(meta, INDEX=index, PAYLOAD=None if index is None else base64.b64encode(chunks[index]).decode())
        code = ("\n".join(f"{key}={value!r}" for key, value in values.items()) + "\n" + RECEIVER).encode("ascii")
        if len(code) > WIRE_MAX:
            raise ValueError("source packet exceeds existing one-program transport")
        compile(code, "<sealed-glm-native-stage>", "exec")
        result.append(code)
    return result
