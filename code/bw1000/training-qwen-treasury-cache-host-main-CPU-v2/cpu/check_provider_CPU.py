"""Cover independent module, distribution and HIP checks with no Torch installation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

from torch_provider import DIST, HIP, MODULE, MODULE_FILE, validate


def main():
    identity = dict(file=MODULE_FILE, distribution_file=MODULE_FILE, version=MODULE,
                    distribution_version=DIST, HIP_build=HIP, SHA256="0"*64)
    validate(identity)
    invalid = dict(version=DIST, distribution_version=MODULE, HIP_build="6.3",
                   file="/tmp/torch/__init__.py", distribution_file="/tmp/torch/__init__.py", SHA256="g"*64)
    for key, value in invalid.items():
        try:
            validate(dict(identity, **{key: value}))
        except ValueError:
            continue
        raise AssertionError(f"bad {key} accepted")
    print("Torch identity: 1 accepted and 6 rejected without device imports")


if __name__ == "__main__":
    main()
