"""Prepare carrier data bytes locally, without contact with the target."""

import argparse
from base64 import b64encode
import json
from pathlib import Path
import re
import runpy


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if HERE != REPO/'code'/'bw1000'/'training-vggt-household-native-input-v2':
    raise RuntimeError('USE_CANONICAL_CHECKOUT_PATH')
ARCHIVE = REPO/'results/bw1000/2026-11-26/training-vggt-household-original3796-zero-GRES-cancelled-terminal.json'
EXISTING = REPO/'code/bw1000/training-vggt-household-native-input-v1'


def assemble(mode, program, api, publication=None, authority=None, intent=None, receipt=None):
    scope = api['REQUEST']
    request = {'check': api['CHECK_REQUEST'], 'launch': scope, 'observe': api['READ_REQUEST']}[mode]
    value = {'request': request, 'mode': mode}
    if mode != 'check':
        if not isinstance(publication,str) or not re.fullmatch('[0-9a-f]{40}', publication):
            raise ValueError('ACTUAL_PUBLICATION_REQUIRED')
        value.update(publication=publication, program_SHA256=api['pin'](program)['SHA256'])
    if mode == 'launch':
        if not isinstance(authority,dict) or authority.get('request') != scope or authority.get('publication') != publication:
            raise ValueError('BIND_ROOT_AUTHORITY_TO_THIS_PUBLICATION_AND_SCOPE')
        value.update(authority=authority,program=b64encode(program).decode('ascii'))
    if mode == 'observe':
        value.update(scope_request=scope,intent_SHA256=intent,receipt_SHA256=receipt)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','launch','observe'))
    parser.add_argument('output',type=Path)
    parser.add_argument('--publication')
    parser.add_argument('--authority-json',type=Path)
    parser.add_argument('--intent-sha256')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args()
    program = (HERE/'carrier.py').read_bytes()
    api = runpy.run_path(str(HERE/'carrier.py'))
    source_raw = (HERE/'SOURCE_FILES.json').read_bytes()
    scope = json.loads((HERE/'CLAIM.json').read_bytes())
    if api['pin'](source_raw) != scope['source_manifest']:
        raise ValueError('SOURCE_MANIFEST_PIN_MISMATCH')
    sources = json.loads(source_raw)['files']
    if set(sources) != {'carrier.py','prepare.py','check_local.py'} or any(
        api['pin']((HERE/name).read_bytes()) != expected for name,expected in sources.items()
    ):
        raise ValueError('PUBLISHED_SOURCE_PIN_MISMATCH')
    claim = json.loads((EXISTING/'CLAIM.json').read_bytes())
    archive_raw = ARCHIVE.read_bytes()
    archive = json.loads(archive_raw)
    assert api['pin'](archive_raw)['SHA256'] == api['ABORT_ARCHIVE'] and archive['replay_allowed'] is False
    assert archive['independent_terminal']['original_receipt_accounting'] == [api['OLD_ROW']]
    for name,expected in claim['files'].items():
        assert api['pin']((EXISTING/name).read_bytes()) == expected
    assert len({api[k] for k in ('REQUEST','CHECK_REQUEST','READ_REQUEST','OLD_REQUEST')}) == 4
    authority = json.loads(args.authority_json.read_bytes()) if args.authority_json else None
    data = assemble(args.mode,program,api,args.publication,authority,args.intent_sha256,args.receipt_sha256)
    encoded = api['pack'](data)
    wire = b64encode(api['pack']({'program':program.decode('utf-8'),'data':b64encode(encoded).decode('ascii')}))
    if not re.fullmatch('[0-9a-f]{32}',data['request']) or len(wire)>560000:
        raise ValueError('TRANSPORT_REQUEST_OR_EXTENT')
    with args.output.open('xb') as file:
        file.write(encoded)
    print(json.dumps({'request':data['request'],'program':api['pin'](program),'data':api['pin'](encoded),
                      'wire_b64_bytes':len(wire),'argv':api['command']()},sort_keys=True))


if __name__ == '__main__':
    main()
