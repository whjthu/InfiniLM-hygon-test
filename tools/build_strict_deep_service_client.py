"""Materialize a fresh DeepSeek client package with strict SSE completion."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARENT = 'code/bw1000/public-bounded-queue-r481/deployment-v1'
R480 = 'code/bw1000/public-token-clock-r480'
FILES = {
    f'{PARENT}/finite_client.py': 'bd568fb0ca273a0f81892d4474784a32f6cadbf352f65ecff184805da7f0fad5',
    f'{PARENT}/scoring.py': '5965c7282e005f60c12cc0f2f719bafa9a87f084e748eb2d148e7ad27d53428d',
    f'{R480}/client_clock.py': 'c1fd4ff49ab222472b1ec6be054f939865f131094ec7e6afa7b10590f9b30ffa',
    f'{R480}/token_clock.py': '086d9ba47efe965a856d4e570c8a7f3f965f58fe203141aa8821383d45be8be6',
    'tools/strict_native_token_clock.py': 'e86e56a76393f51cbadcc37dbbe79b7db0eb4e8a3b0c13a9df49a5b30b7fe13f',
}
CONTRACT = 'infinilm-native-token-clock-strict-wire-v1'
EDITS = (
    ('# Passed frozen r480 source directory must already be on PYTHONPATH.\nfrom client_clock import ClientClock',
     'from strict_native_token_clock import ClientClock, verify_dependencies'),
    ('def main(config):\n', 'def main(config):\n verify_dependencies()\n'),
    ('result=dict(completed=False,planned=20', f'result=dict(clock_contract={CONTRACT!r},completed=False,planned=20'),
    ("if not b:r['fully_received_EOF']=True;break",
     "if not b:r['fully_received_EOF']=clock.finish_http_body(response);break"),
    ('r.update(completed=True,clock=clock.summary(),frames=clock.frames,read_clock=reads)',
     "clock_summary=clock.summary();r.update(completed=clock_summary['complete'],clock=clock_summary,frames=clock.frames,read_clock=reads)"),
    ("natural=bool(clock.get('EOS_tokens')==1", "natural=bool(clock.get('complete') is True and clock.get('EOS_tokens')==1"),
)


def render(source):
    text = source.decode('utf-8')
    for before, after in EDITS:
        if text.count(before) != 1:
            raise ValueError('finite client integration boundary differs')
        text = text.replace(before, after, 1)
    ast.parse(text, filename='finite_client.py', feature_version=(3, 10))
    return text.encode('utf-8')


def materialize(repo, output):
    files = {}
    for relative, digest in FILES.items():
        raw = (repo / relative).read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f'frozen client dependency differs: {relative}')
        files[Path(relative).name] = raw
    files['finite_client.py'] = render(files['finite_client.py'])
    receipt = dict(schema=1, clock_contract=CONTRACT, original_files_SHA256=FILES,
                   files_SHA256={name: sha256(data).hexdigest() for name, data in sorted(files.items())},
                   service_or_GPU_authorized=False)
    output.mkdir(mode=0o700)
    for name, content in files.items():
        with (output / name).open('xb') as stream:
            stream.write(content)
    with (output / 'CLIENT_SOURCE_MANIFEST.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(materialize(REPO, parser.parse_args().output)))


if __name__ == '__main__':
    main()
