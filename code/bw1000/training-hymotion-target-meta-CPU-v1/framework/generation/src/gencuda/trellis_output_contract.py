"""Fail-closed output contracts for bounded TRELLIS inference."""
from __future__ import annotations

from typing import Any, Mapping


LATENT_ONLY = "latent_only"
DIAGNOSTIC_MULTIBOX = "diagnostic_multibox"
FULL_MESH = "full_mesh"
SUPPORTED_OUTPUT_CONTRACTS = frozenset({LATENT_ONLY, DIAGNOSTIC_MULTIBOX, FULL_MESH})


def validate_output_contract(config: Mapping[str, Any]) -> str:
    """Validate the explicit output contract before runtime setup."""
    value = config.get("output_contract")
    if type(value) is not str or value not in SUPPORTED_OUTPUT_CONTRACTS:
        raise ValueError(
            "TRELLIS output_contract must be exactly 'latent_only' or "
            "'diagnostic_multibox' or explicitly configured 'full_mesh'"
        )
    from .trellis_sparse_support import validate_limit
    if value == FULL_MESH and "max_sparse_tokens" not in config:
        raise ValueError("full_mesh requires an explicit max_sparse_tokens admission budget")
    validate_limit(config.get("max_sparse_tokens", 1024))
    decode_cpu_fields = config.get("decode_cpu_fields", False)
    if type(decode_cpu_fields) is not bool:
        raise ValueError("TRELLIS decode_cpu_fields must be a boolean")
    if decode_cpu_fields and value not in {LATENT_ONLY, FULL_MESH}:
        raise ValueError("TRELLIS decode_cpu_fields is only supported with latent_only or full_mesh")
    if value == FULL_MESH:
        if not decode_cpu_fields:
            raise ValueError("full_mesh requires decode_cpu_fields=true")
        from .trellis_cpu_texture_export import settings
        settings(config.get("cpu_texture_export"))
    elif "cpu_texture_export" in config:
        raise ValueError("cpu_texture_export requires full_mesh")
    source_root = config.get("cpu_decoder_source_root")
    if source_root is not None and (type(source_root) is not str or not source_root):
        raise ValueError("TRELLIS cpu_decoder_source_root must be a non-empty string")
    for name in (
        "cpu_decode_max_points",
        "cpu_decode_threads",
        "cpu_decode_timeout_seconds",
    ):
        setting = config.get(name)
        if setting is not None and (type(setting) is not int or setting < 1):
            raise ValueError(f"TRELLIS {name} must be a positive integer")
    resolution = config.get("cpu_decode_resolution")
    if decode_cpu_fields and resolution is None:
        raise ValueError("TRELLIS cpu_decode_resolution is required for CPU decoding")
    if resolution is not None and (
        type(resolution) is not int or resolution not in {256, 512, 1024}
    ):
        raise ValueError("TRELLIS cpu_decode_resolution must be 256, 512, or 1024")
    # The staged generation backend currently samples only the 512 flow branch.
    if decode_cpu_fields and resolution != 512:
        raise ValueError("TRELLIS staged generation requires cpu_decode_resolution=512")
    for name in ("cpu_shape_decoder_sha256", "cpu_material_decoder_sha256"):
        digest = config.get(name)
        if digest is not None and (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"TRELLIS {name} must be a lowercase SHA256 digest")
    return value
