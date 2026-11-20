"""Rebind only the known model-provider bytes in the original Qwen input contract."""
import hashlib
import json

from build_overlay import SOURCE_SHA256

ORIGINAL_BINDING_SHA256 = "9c9ba809529c73098e71f93ad1bf4744cb84079160f7062a540e7a37cb144067"
PATCHED_PROVIDER_SHA256 = "40e50bc545325c01bc2a50c324ba161dc1d29aef04d09c58b73f859208f76013"


def bound_bytes(original, mounted_provider):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_BINDING_SHA256:
        raise ValueError("unexpected original training binding")
    if hashlib.sha256(mounted_provider).hexdigest() != PATCHED_PROVIDER_SHA256:
        raise ValueError("unexpected actual target provider")
    initial = json.loads(original)
    if initial["model_source_SHA256"] != SOURCE_SHA256 or original.count(SOURCE_SHA256.encode()) != 1:
        raise ValueError("ambiguous original provider identity")
    patched = original.replace(SOURCE_SHA256.encode(), PATCHED_PROVIDER_SHA256.encode(), 1)
    expected = {**initial, "model_source_SHA256": PATCHED_PROVIDER_SHA256}
    if json.loads(patched) != expected:
        raise ValueError("other original training binding fields changed")
    return patched


def write_binding(original, target, provider):
    patched = bound_bytes(original.read_bytes(), provider)
    with target.open("xb") as stream:
        stream.write(patched)
    return hashlib.sha256(patched).hexdigest()
