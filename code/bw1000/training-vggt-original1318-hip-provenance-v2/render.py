"""Change only the HIP evidence call and module in an exact .1318 controller render."""

from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_LAUNCH_SHA = "1e29f43be1b36e7db146c58ca0da0f854d6ee2b06e9949bd1fb1219dc043d859"
BASE_DSO_SHA = "e55b9f73f20f64edf3fc6ee22925c8c732a9555f2cad9c4458a1cc243b19bad1"
DSO = "tools/vggt_1318_hip_provenance.py"
OLD_REQUEST = "15fe088576824de1a3e6b7fc77b305f5"
FAILED_SOURCE_STATE = "ORIGINAL_3188_NO_PROVIDER_ATTESTATION"


def digest(raw):
    return sha256(raw).hexdigest()


def render(original, scope, read=None):
    if digest(Path(original.__file__).read_bytes()) != BASE_LAUNCH_SHA:
        raise ValueError("original .1318 launcher bytes changed")
    if scope["request"] == OLD_REQUEST:
        raise ValueError("original .3188 request is closed")
    sources, prior, label = original.render(scope) if read is None else original.render(scope, read)
    controller = original.CONTROLLER
    index = original.auth.LEGACY_INDEX
    previous = json.loads(sources[index])
    if digest(sources[DSO]) != BASE_DSO_SHA or previous[DSO] != BASE_DSO_SHA:
        raise ValueError("original .1318 HIP evidence source changed")
    source = sources[controller]
    for old in (b"observed_so = hip_dso()", b"require(hip_dso() == observed_so"):
        if source.count(old) != 1:
            raise ValueError("original .1318 HIP handle anchor changed")
        source = source.replace(old, old.replace(b"hip_dso()", b"hip_dso(hip)"), 1)
    replacement = (HERE / "hip_provenance.py").read_bytes()
    compile(replacement, DSO, "exec")
    compile(source, controller, "exec")
    change = {controller: source, DSO: replacement}
    new_index = {**previous, **{name: digest(raw) for name, raw in change.items()}}
    return {**sources, **change, index: original.auth.encode(new_index)}, prior, label
