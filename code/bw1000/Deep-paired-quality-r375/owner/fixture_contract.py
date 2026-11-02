"""Pinned existing frozen real fixture; its body is not copied into this source package."""
import pathlib,json,hashlib
D=pathlib.Path(__file__).resolve().parent
def load():
    ref=json.loads((D/'FROZEN_FIXTURE_REF.json').read_bytes());p=pathlib.Path(ref['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['SHA256']
    q=json.loads(p.read_bytes());assert q['actual_prompt_tokens']==ref['actual_prompt_tokens']==8489
    assert q['actual_prompt_token_ids_SHA256']==ref['actual_prompt_token_ids_SHA256'];return q
