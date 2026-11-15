"""Check the exact HunyuanVideo SFT replay inputs before reserving a DCU."""
import argparse
import hashlib
import json
import re
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
SUFFIX_TRANSFER = '0d16553646bf4dad8cdf74209a78b7d2'
SUFFIX_CLAIM = '855e5f79de8591dd707f0b072a203716d7ddd105'
SUFFIX_EPOCH = 1794704140.3119218
SUFFIX_BYTES = 12341840927
PREFIX_OFFSET = 20964791265
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



def final_suffix_identity(root, audit_path, audit_sha):
    if not isinstance(audit_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", audit_sha):
        raise ValueError("Expected the published audit SHA-256")
    with audit_path.open("rb") as f:
        raw = f.read(65537)
    if not 0 < len(raw) <= 65536 or hashlib.sha256(raw).hexdigest() != audit_sha:
        raise ValueError("Published audit bytes differ")
    audit = json.loads(raw)
    proof = audit.get("result", {})
    if (audit.get("passed") is not True or audit.get("remote_outcome") != "VERIFIED"
            or proof.get("passed") is not True or proof.get("request_id") != SUFFIX_TRANSFER
            or proof.get("claim") != SUFFIX_CLAIM or proof.get("weight_SHA256") != WEIGHT_SHA
            or proof.get("config_SHA256") != CONFIG_SHA or proof.get("bytes") != WEIGHT_SIZE
            or proof.get("GPU_calls") != 0):
        raise ValueError("Independent audit differs from the original Transformer suffix")
    final = root / "hunyuanvideo-1.5"
    folder = final / "transformer/480p_t2v"
    control = root / "a100-verified-transformer-suffix-v1"
    if (any(p.is_symlink() or not p.is_dir() for p in (root, final, folder.parent, folder, control))
            or (root / "model.incomplete").exists() or (root / "model.incomplete").is_symlink()):
        return {"state": "FINAL_PATH_MISMATCH"}
    receipt = control / "outcome.json"
    with receipt.open("rb") as f:
        raw = f.read(16385)
    digest = hashlib.sha256(raw).hexdigest()
    if not 0 < len(raw) <= 16384 or digest != proof.get("terminal_SHA256"):
        return {"state": "RECEIVER_TERMINAL_MISMATCH"}
    expected = dict(claim=SUFFIX_CLAIM, request_id=SUFFIX_TRANSFER, epoch=SUFFIX_EPOCH, role="receiver",
                    passed=True, suffix_bytes=SUFFIX_BYTES, files={"diffusion_pytorch_model.safetensors":
                    dict(bytes=WEIGHT_SIZE, offset=PREFIX_OFFSET, SHA256=WEIGHT_SHA)})
    if json.loads(raw) != expected:
        return {"state": "RECEIVER_TERMINAL_MISMATCH"}
    info = (folder / "diffusion_pytorch_model.safetensors").lstat()
    stamp = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    if not stat.S_ISREG(info.st_mode) or stamp != proof.get("file_identity"):
        return {"state": "AUDITED_FILE_IDENTITY_CHANGED"}
    return dict(state="VERIFIED_RECEIVER_IDENTITY", final_bytes=info.st_size,
                receipt_SHA256=digest, independent_audit_SHA256=audit_sha)


def inspect(results, audit_path=None, audit_sha=None):
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
    weight = (final_suffix_identity(results / MODEL, audit_path, audit_sha)
              if audit_path is not None else weight_receipt(results / MODEL))
    return dict(case='T09.sft', observed_epoch=time.time(), files=files, transformer=weight,
                input_identity_ready=all(row['state'] == 'VERIFIED' for row in files.values())
                and weight['state'] == 'VERIFIED_RECEIVER_IDENTITY',
                transformer_payload_bytes_read=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument("--transformer-audit", type=Path)
    parser.add_argument("--transformer-audit-sha256")
    args = parser.parse_args()
    if (args.transformer_audit is None) != (args.transformer_audit_sha256 is None):
        parser.error("supply both Transformer audit arguments")
    print(json.dumps(inspect(args.results, args.transformer_audit, args.transformer_audit_sha256), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
