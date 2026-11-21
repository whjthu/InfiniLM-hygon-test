"""Dispatch the frozen GLM I128 component through the published depot controller."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CLAIM = 'coordination/tasks/glm-i128-four-routed-variants-device-after-link-v2.json'
ASSEMBLER = ('code/bw1000/GLM-C1-layer-observer-r474/'
             'routed-i128-component-depot-v2/launch.py')
ASSEMBLER_SHA = 'ff131982171711a482208851aebe4c86e2380e07f6a221fbae6e4ccd83b64a04'
REQUEST = '6f808bade51aded8234e2dbaee8b2af5'
LINK_RECEIPT_SHA = '3afb5d888ea0fb74aefeded292469c3167ded68ed5c2da1a39497e2e85f05ce4'
INTENT = 'local-only/GLM-I128-four-routed-variants-one-device-v1/intent.json'


def main(publication, transport):
    path = REPO / ASSEMBLER
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != ASSEMBLER_SHA:
        raise ValueError('the original depot assembler changed')
    spec = importlib.util.spec_from_file_location('glm_i128_depot_assembler', path)
    assembler = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), assembler.__dict__)
    claim = json.loads(assembler.published_blob(publication, CLAIM))
    if (claim['request'] != REQUEST or claim['GPU_GO'] is not True
            or claim['link_report_SHA256'] != LINK_RECEIPT_SHA):
        raise ValueError('scope or original link receipt changed')
    body = assembler.packet(publication, 'component', CLAIM, link_sha=LINK_RECEIPT_SHA)
    intent = dict(request=REQUEST, publication=publication,
                  body_SHA256=sha256(body).hexdigest(), parent='1103606',
                  node='f13r1n08', DCU=1, replay_allowed=False)
    return transport(body, intent)
