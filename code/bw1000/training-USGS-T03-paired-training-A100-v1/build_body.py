"""Root-only local payload preparation; ordinary SSH dispatch stays with coordinator."""
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import time

D = Path(__file__).resolve().parent


def need(ok, code):
    if not ok:
        raise ValueError(code)


def make_body(arm, publication, request_id, epoch, domain_qualification=None):
    def check():
        need(type(epoch) in (int, float) and math.isfinite(epoch) and 0 < epoch <= time.time() < epoch + 740,
             'CLIENT_ORIGINAL900_EPOCH_EXHAUSTED')
    check()
    need(arm in ('domain', 'generic') and re.fullmatch('[0-9a-f]{40}', publication) and
         re.fullmatch('[0-9a-f]{32}', request_id), 'ONE_PUBLISHED_ARM_CONTEXT')
    mf = (D / 'SOURCE_FILES.json').read_bytes(); files = json.loads(mf)['files']
    actual = {p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file() and p != D / 'SOURCE_FILES.json'}
    need(set(files) == actual, 'EXACT_SOURCE_INVENTORY')
    package = {'SOURCE_FILES.json': base64.b64encode(mf).decode()}
    for name, pin in files.items():
        check(); path = D / name; need(not path.is_symlink(), 'SOURCE_SYMLINK')
        raw = path.read_bytes()
        need(len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256'], 'SOURCE_SHA')
        package[name] = base64.b64encode(raw).decode()
    raw_claim = (D / 'CLAIM.json').read_bytes(); claim = json.loads(raw_claim)
    runpy.run_path(str(D / 'verify_package.py'), run_name='package_verification')['verify'](D)
    need(claim['root_GO'] is True and claim['GPU_GO'] is True and claim['max_invocations_per_arm'] == 1,
         'ROOT_ACTUAL_ADOPTION_REQUIRED')
    payload = dict(arm=arm, publication=publication, request_id=request_id, epoch=epoch, package=package,
        source_manifest_SHA256=hashlib.sha256(mf).hexdigest(), claim_SHA256=hashlib.sha256(raw_claim).hexdigest())
    if arm == 'generic':
        need(isinstance(domain_qualification, bytes) and 0 < len(domain_qualification) <= 16384,
             'ACTUAL_DOMAIN_INDEPENDENT_RECEIPT_REQUIRED')
        q = json.loads(domain_qualification)
        need(q['schema'] == 'T03_PAIRED_DOMAIN25_INDEPENDENT_TERMINAL_V1' and q['independent_passed'] is True and
             q['engineering_completed'] is True and q['release_permitted'] is True and q['arm'] == 'domain' and
             q['case'] == 'T03.sft' and q['pair_id'] == claim['pair_id'] and q['completed_ranks'] == 8 and
             q['completed_updates'] == 25 and q['global_micro_exposures'] == 800 and
             q['source_manifest_SHA256'] == payload['source_manifest_SHA256'], 'DOMAIN_FIRST_CONTROL_NOT_READY')
        payload['domain_qualification_b64'] = base64.b64encode(domain_qualification).decode()
    else:
        need(domain_qualification is None, 'DOMAIN_HAS_NO_FAKE_PRIOR_RESULT')
    encoded = json.dumps(payload, separators=(',', ':')).encode()
    need(len(encoded) < 3 * 1024**2, 'ROOT_PAYLOAD_BOUND')
    check()
    return 'INSTALL_PAYLOAD_B64=' + repr(base64.b64encode(encoded).decode()) + '\n' + (D / 'install_and_run.py').read_text()
