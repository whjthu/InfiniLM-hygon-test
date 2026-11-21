"""Acquire one non-overlapping TUM recording pair; no full archive or retry."""
import hashlib
import importlib.util
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

G = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = G / 'local-only/training-vggt-tum-sitting-xyz-network-recovery-v2'
HELPER_SHA256 = '3bcb569c44ab2a33d7b53f843080c765706e2c67c79166df6ac96255214da3e0'
SEQUENCE = 'freiburg3_sitting_xyz'
RESERVED = {
    'train': ['freiburg3_structure_texture_near', 'freiburg3_nostructure_texture_far'],
    'dev': ['freiburg3_cabinet', 'freiburg3_teddy', 'freiburg1_plant', 'freiburg2_dishes'],
    'blind': ['freiburg3_long_office_household'],
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def store(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(raw)
    temp.replace(path)


def main():
    origin = time.time()
    end = time.monotonic() + 180
    OUT.mkdir()
    report = {'sequence': SEQUENCE, 'origin': origin, 'deadline': origin + 180,
              'reserved_recordings': RESERVED, 'requests': [], 'actual_read_bytes': 0,
              'planned_pairs': 1, 'completed_pairs': 0, 'GPU_calls': 0, 'updates': 0,
              'train_GO': False, 'READY': False}
    store('intent.json', report)
    try:
        assert all(SEQUENCE not in values for values in RESERVED.values())
        helper = HERE / 'frozen_pair_acquisition.py'
        assert digest(helper.read_bytes()) == HELPER_SHA256
        spec = importlib.util.spec_from_file_location('frozen_tum_acquisition', helper)
        h = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(h)
        h.SCENE = 'sitting_xyz'
        h.SEQUENCE = SEQUENCE
        h.ARCHIVE_ROOT = 'rgbd_dataset_' + SEQUENCE
        h.TUM = 'https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/' + h.ARCHIVE_ROOT
        h.MREL = 'data/freiburg3/fr3_sitting_xyz.fo.mcap'
        h.MCAP = h.HF + 'resolve/' + h.REV + '/' + h.MREL
        h.R = OUT

        class Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                u = urllib.parse.urlsplit(newurl)
                if u.scheme != 'https' or not (u.hostname == 'huggingface.co' or
                        u.hostname.endswith('.huggingface.co') or u.hostname.endswith('.hf.co')):
                    raise ValueError('Redirect outside frozen public source asset domains')
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(Redirect())

        def request(url, cap, byte_range=None):
            if len(report['requests']) >= 20 or time.monotonic() >= end:
                raise TimeoutError('Source acquisition budget exhausted')
            urlid = digest(url.encode())
            identity = (urlid, byte_range)
            if any((r['URL_SHA256'], r['range']) == identity for r in report['requests']):
                raise ValueError('Duplicate request prohibited')
            row = {'URL_SHA256': urlid, 'range': byte_range, 'attempted': True,
                   'read_bytes': 0, 'status': None, 'EOF': False}
            report['requests'].append(row)
            store('progress.json', report)
            headers = {'Accept-Encoding': 'identity'}
            if byte_range:
                headers['Range'] = byte_range
            try:
                with opener.open(urllib.request.Request(url, headers=headers),
                                 timeout=min(20, end-time.monotonic())) as response:
                    row['status'] = response.status
                    if byte_range:
                        m = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                         response.headers.get('Content-Range', ''))
                        if response.status != 206 or m is None:
                            raise ValueError('Exact Range not served; body not read')
                        first, last, total = map(int, m.groups())
                        if first == 0 and last == total-1 or last-first+1 > cap:
                            raise ValueError('Full object or excessive returned range')
                        if byte_range not in (f'bytes={first}-{last}', f'bytes=-{last-first+1}'):
                            raise ValueError('Mismatched Range; body not read')
                        cap = last-first+1
                    elif response.status != 200:
                        raise ValueError('Non-success metadata response')
                    body_length = response.headers.get('Content-Length')
                    if body_length is not None:
                        if not re.fullmatch(r'[0-9]{1,10}', body_length) or not 0 < int(body_length) <= cap:
                            raise ValueError('Invalid bounded Content-Length')
                        body_length = int(body_length)
                        if byte_range and body_length != cap:
                            raise ValueError('Range and Content-Length disagree')
                    if report['actual_read_bytes']+cap+1 > 18*1024*1024:
                        raise ValueError('Global source bytes cap')
                    pieces = []
                    while row['read_bytes'] <= cap:
                        if response.isclosed():
                            row['EOF'] = True
                            break
                        left = end-time.monotonic()
                        if left <= 0:
                            raise TimeoutError('Original epoch exhausted')
                        try:
                            response.fp.raw._sock.settimeout(min(20, left))
                        except AttributeError:
                            raise ValueError('Cannot enforce remaining read timeout')
                        raw = response.read(min(65536, cap+1-row['read_bytes']))
                        if not raw:
                            row['EOF'] = True
                            break
                        pieces.append(raw)
                        row['read_bytes'] += len(raw)
                        report['actual_read_bytes'] += len(raw)
                    body = b''.join(pieces)
                    expected_length = cap if byte_range else body_length
                    if not body or len(body) > cap or (expected_length is not None and len(body) != expected_length):
                        raise ValueError('Invalid bounded body length')
                    row['SHA256'] = digest(body)
                    piece = OUT / ('HTTP-%02d.private.bin' % len(report['requests']))
                    with piece.open('xb') as saved:
                        saved.write(body)
                    row['saved_name'] = piece.name
                    info = {'url': url, 'bytes': len(body), 'SHA256': digest(body)}
                    if byte_range:
                        info.update(start=first, end=last, total=total)
                    h.REPORT.setdefault('retrievals', []).append(info)
                    store('SOURCE.json', h.REPORT)
                    return body, info
            except Exception as error:
                row['failure_type'] = type(error).__name__
                raise
            finally:
                store('progress.json', report)

        h.request = request
        # Freeze the MCAP identity and index before fetching any original RGB prefix.
        summary = h.summary()
        h.summary = lambda: summary
        h.acquire()
        store('SOURCE.json', h.REPORT)
        report['completed_pairs'] = 1
        report['state'] = 'SOURCE_PAIR_PINNED_GEOMETRY_PENDING'
    except Exception as error:
        report.update(state='SOURCE_PAIR_INCOMPLETE', error_type=type(error).__name__,
                      error_message=str(error)[:256])
    report['wall_seconds'] = time.time()-origin
    store('result.json', report)
    print(json.dumps({k: report[k] for k in ('state', 'completed_pairs', 'planned_pairs',
          'actual_read_bytes', 'wall_seconds', 'READY')}, sort_keys=True))
    return 0 if report['completed_pairs'] == 1 else 1


if __name__ == '__main__':
    raise SystemExit(main())
