"""Use the qualified eager OneVision import guard with the pinned target runtime."""

from hashlib import sha256
from pathlib import Path


PLUGIN_SHA256 = "fbfd1875cb169c3b1bc4862d4d2bf94317ba92a8a88e31f9419a1e91a4e2532c"
MODEL_SHA256 = "bb7ee743a32a2aa03789445794fe328ea21edb9b6e9f110d4bc5576f0ad29198"
ORIGINAL_PROVIDER = "5.12.1"
TARGET_PROVIDER = "5.5.0"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_file(path, cap, expected):
    with path.open("rb") as source:
        raw = source.read(cap + 1)
    require(0 < len(raw) <= cap and sha256(raw).hexdigest() == expected, "Exact eager source identity differs")


def preimport(root, compatibility, recovered):
    """Extend only the checked import guard's version for one eager model import."""
    import transformers
    from transformers import modeling_flash_attention_utils as flash_utils

    require(transformers.__version__ == TARGET_PROVIDER and not hasattr(flash_utils, "flash_attn_varlen_func"),
            "Unexpected target Transformers capability")
    require(compatibility._PINNED_TRANSFORMERS_VERSION == ORIGINAL_PROVIDER
            and compatibility._TARGET_SOURCE_SHA256 == MODEL_SHA256, "Unexpected original OneVision guard")
    check_file(Path(compatibility.__file__), 65536, PLUGIN_SHA256)
    check_file(root / compatibility._TARGET_SOURCE_NAME, 262144, MODEL_SHA256)
    original = compatibility._preimport_verified_eager_class
    predicate = flash_utils.is_flash_attn_available
    qualified = None

    def verify_preimport(path, config, options):
        nonlocal qualified
        require(qualified is None and Path(path).resolve() == root.resolve()
                and options == {"attn_implementation": "eager"}
                and compatibility._explicit_eager_onevision(config, options),
                "The compatibility window is restricted to the pinned eager OneVision class")
        qualified = original(path, config, options)
        require(isinstance(qualified, str) and qualified.endswith(".modeling_llavaonevision1_5"),
                "The original guarded import did not return the expected native module")
        return qualified

    with compatibility._EAGER_PREIMPORT_LOCK:
        try:
            compatibility._PINNED_TRANSFORMERS_VERSION = TARGET_PROVIDER
            compatibility._preimport_verified_eager_class = verify_preimport
            cls = recovered.preimport_verified_class(root, compatibility)
        finally:
            compatibility._preimport_verified_eager_class = original
            compatibility._PINNED_TRANSFORMERS_VERSION = ORIGINAL_PROVIDER
    require(qualified == cls.__module__ and flash_utils.is_flash_attn_available is predicate
            and not hasattr(flash_utils, "flash_attn_varlen_func"), "Eager import could not safely restore runtime state")
    return cls
