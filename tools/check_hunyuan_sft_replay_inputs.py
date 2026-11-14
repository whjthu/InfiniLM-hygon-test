"""Check the exact HunyuanVideo SFT replay inputs before reserving a DCU."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import time

CACHE = 'training-hunyuanvideo-original-inputs-v1'
REPLAY = 'training-hunyuanvideo-first-step-replay-a100-v1'
MODEL = 'training-hunyuanvideo-original-model-v1'
TRANSFER = '0a610475811d4237955e139ab9dda71c'
CLAIM = 'e124758359448cf6ff5d05ed4141783b7a745bf8'
TRANSFER_DEADLINE = 1794681036.3827512
WEIGHT_SIZE = 33306632192
WEIGHT_SHA = '71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86'
CONFIG_SHA = '689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae'


def small_file(path, expected_size, expected_sha):
    try:
        with path.open('rb') as stream:
            raw = stream.read(expected_size + 1)
    except FileNotFoundError:
        return {'state': 'MISSING'}
    digest = hashlib.sha256(raw).hexdigest()
    return dict(state='VERIFIED' if (len(raw), digest) == (expected_size, expected_sha)
                else 'IDENTITY_MISMATCH', bytes_read=len(raw), SHA256=digest)


def weight_receipt(root):
    receipt = root / 'retained-suffix-v2/outcome.json'
    try:
        with receipt.open('rb') as stream:
            raw = stream.read(16385)
    except FileNotFoundError:
        return {'state': 'TERMINAL_RECEIPT_MISSING'}
    if len(raw) > 16384:
        return {'state': 'INVALID_TERMINAL_RECEIPT'}
    report = {'receipt_SHA256': hashlib.sha256(raw).hexdigest()}
    try:
        saved = json.loads(raw)
    except (ValueError, UnicodeError):
        return dict(report, state='INVALID_TERMINAL_RECEIPT')
    if (type(saved) is not dict or saved.get('request_id') != TRANSFER or saved.get('claim') != CLAIM
            or saved.get('deadline') != TRANSFER_DEADLINE or type(saved.get('passed')) is not bool):
        return dict(report, state='INVALID_TERMINAL_RECEIPT')
    if not saved['passed']:
        return dict(report, state='RECEIVER_REPORTED_FAILURE')
    final = root / 'hunyuanvideo-1.5/transformer/480p_t2v/diffusion_pytorch_model.safetensors'
    try:
        info = final.stat()
    except FileNotFoundError:
        return dict(report, state='FINAL_FILE_MISSING')
    qualifies = (saved.get('offset') == WEIGHT_SIZE and saved.get('whole_SHA256') == WEIGHT_SHA
                 and stat.S_ISREG(info.st_mode) and info.st_size == WEIGHT_SIZE)
    return dict(report, state='VERIFIED_RECEIVER_IDENTITY' if qualifies else 'FINAL_IDENTITY_MISMATCH',
                final_bytes=info.st_size)


def inspect(results):
    files = {
        'sample.pt': small_file(results / CACHE / 'sample.pt', 3521630,
                                '9bdbaccaada6be6080ace96aa464f70adc8041bd43e58eae58052c5b2eab541b'),
        'staged_config': small_file(results / CACHE / 'config.json', 968, CONFIG_SHA),
        'native_model_config': small_file(
            results / MODEL / 'hunyuanvideo-1.5/transformer/480p_t2v/config.json', 968, CONFIG_SHA),
        'before.pt': small_file(results / REPLAY / 'before.pt', 7149635,
                                '2856fae7918e67fcda1abf4b9ec0106f8fe11900132397ab71c65d9ed5f1ac46'),
        'inputs.pt': small_file(results / REPLAY / 'inputs.pt', 1853149,
                                '5ee197e410ca7a5595b98375be64d392028f466f249a222aea532f2841a1851c'),
    }
    weight = weight_receipt(results / MODEL)
    return dict(case='T09.sft', observed_epoch=time.time(), files=files, transformer=weight,
                input_identity_ready=all(row['state'] == 'VERIFIED' for row in files.values())
                and weight['state'] == 'VERIFIED_RECEIVER_IDENTITY',
                transformer_payload_bytes_read=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.results), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
