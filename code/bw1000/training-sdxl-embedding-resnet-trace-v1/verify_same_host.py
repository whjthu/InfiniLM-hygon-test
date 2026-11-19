"""Exact fingerprint-free comparison of same-platform original and traced updates."""
import hashlib
import json
from pathlib import Path

import torch


FIELDS = ('initial_adapter','initial_optimizer','inputs','batch','target','raw_prediction',
          'prediction','loss','preclip_norm','gradients_before_clip','gradients_after_clip',
          'final_adapter','final_optimizer')


def compare(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape
                and left.dtype == right.dtype and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(compare(value, right[k]) for k,value in left.items())
    if isinstance(left, (list,tuple)):
        return len(left) == len(right) and all(compare(a,b) for a,b in zip(left,right))
    return left == right


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def verify(old, new, baseline_sha):
    if sha(old) != baseline_sha:
        raise ValueError('BASELINE_ARCHIVE_PIN_CHANGED')
    base = torch.load(old,weights_only=True,map_location='cpu')
    candidate = torch.load(new,weights_only=True,map_location='cpu')
    if any(key not in base or key not in candidate for key in FIELDS):
        raise ValueError('MISSING_SAME_PLATFORM_UPDATE_FIELDS')
    result={field:compare(base[field],candidate[field]) for field in FIELDS}
    return dict(exact=all(result.values()),fields=result,tensors=len(candidate['final_adapter']),
                original_observed_SHA256=baseline_sha,traced_observed_SHA256=sha(new))
