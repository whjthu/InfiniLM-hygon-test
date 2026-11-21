"""Assemble the revised GLM component from the already audited original depot."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]
REL = PACKAGE.relative_to(REPO).as_posix()
AREA = 'code/bw1000/GLM-C1-layer-observer-r474'
ASSEMBLER = f'{AREA}/routed-i128-component-depot-v2/launch.py'
ASSEMBLER_SHA = 'ff131982171711a482208851aebe4c86e2380e07f6a221fbae6e4ccd83b64a04'
CALLER = f'{AREA}/routed-i128-one-gpu-caller-v1'
CONTROL = 'tools/control_glm_i128_component.py'
INDEX = 'coordination/tuning/glm-i128-component-controller-files-v1.json'
OLD_LABEL = 'GLM-I128-four-routed-variants-one-device-v1'
LABEL = 'GLM-I128-four-routed-variants-canonical-HIP-v2'
CLAIM = 'coordination/tasks/glm-i128-four-routed-variants-canonical-hip-v2.json'
INTENT = f'local-only/{LABEL}/intent.json'
LINK_SHA = '3afb5d888ea0fb74aefeded292469c3167ded68ed5c2da1a39497e2e85f05ce4'


def replace_once(raw, previous, updated):
    if raw.count(previous) != 1:
        raise ValueError('the frozen source boundary changed')
    return raw.replace(previous, updated, 1)


def revision(publication):
    path = REPO / ASSEMBLER
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != ASSEMBLER_SHA:
        raise ValueError('the original depot assembler changed')
    spec = importlib.util.spec_from_file_location('glm_i128_verified_depot', path)
    prior = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), prior.__dict__)
    files = prior.depot(publication)
    hip = f'{CALLER}/hip_runtime.py'
    call = f'{CALLER}/caller.py'
    caller_mf = f'{CALLER}/SOURCE_FILES.json'
    files[hip] = replace_once(files[hip], b'def file_pin(item):', b'def file_pin(item, *, exact_dtk_hip=False):')
    old = br'''    if (not path.is_absolute() or not re.fullmatch(r".+\.so(?:\.\d+)*", path.name)
            or not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha)):'''
    new = br'''    allowed_dtk_build = (
        exact_dtk_hip
        and path.name == "libgalaxyhip.so.5.2.26113.2065-6699a1f0"
        and sha == "b125b6b4153f5d25dd8e445b43b2f57b0eab9b6305eb765430a82a4359e1870b"
    )
    normal_library_name = re.fullmatch(r".+\.so(?:\.\d+)*", path.name)
    if (not path.is_absolute() or not (normal_library_name or allowed_dtk_build)
            or not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha)):'''
    files[hip] = replace_once(files[hip], old, new)
    files[call] = replace_once(files[call], b'H.file_pin(raw.get("hip"))',
                               b'H.file_pin(raw.get("hip"), exact_dtk_hip=True)')
    caller = json.loads(files[caller_mf])
    for path in (hip, call):
        caller['files_SHA256'][Path(path).name] = sha256(files[path]).hexdigest()
        compile(files[path], path, 'exec')
    files[caller_mf] = (json.dumps(caller, sort_keys=True) + '\n').encode()
    caller_pin = sha256(files[caller_mf]).hexdigest()
    files[CONTROL] = replace_once(files[CONTROL], OLD_LABEL.encode(), LABEL.encode())
    files[CONTROL] = replace_once(
        files[CONTROL], b'ce395475027bb715363d6b8bc03cd5656d7b2ff401c610592fbbedd091a5df3a', caller_pin.encode())
    compile(files[CONTROL], CONTROL, 'exec')
    index = json.loads(files[INDEX])
    for path in (CONTROL, caller_mf):
        index[path] = sha256(files[path]).hexdigest()
    files[INDEX] = (json.dumps(index, sort_keys=True) + '\n').encode()
    request = sha256((LABEL + ':' + caller_pin).encode()).hexdigest()[:32]
    return prior, files, request, caller_pin


def main(publication, transport):
    prior, files, request, caller_pin = revision(publication)
    raw = prior.published_blob(publication, CLAIM)
    claim = json.loads(raw)
    pins = {name: sha256(data).hexdigest() for name, data in files.items()}
    if (claim['request'] != request or claim['GPU_GO'] is not True or claim['depot'] != pins
            or claim['fixture_only'] is not True or claim['link_report_SHA256'] != LINK_SHA
            or claim['caller_manifest_SHA256'] != caller_pin):
        raise ValueError('revised component scope differs from the published claim')
    receiver = prior.published_blob(publication, REL + '/receive.py')
    payload = dict(publication=publication, claim=raw, files=files)
    body = ('PAYLOAD=' + repr(payload) + '\n').encode() + receiver
    intent = dict(request=request, publication=publication, body_SHA256=sha256(body).hexdigest(),
                  parent='1103606', node='f13r1n08', DCU=1, replay_allowed=False)
    return transport(body, intent)
