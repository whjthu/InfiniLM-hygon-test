"""Verify the actual isolated providers and run the unchanged native import audit."""

import hashlib
import importlib.util
import json
from pathlib import Path
import runpy


def main():
    claim = json.loads(Path("/import-v2/CLAIM.json").read_bytes())
    for name in ("transformers", "huggingface_hub"):
        spec = importlib.util.find_spec(name)
        if spec is None or not Path(spec.origin).resolve().is_relative_to("/zimage-compat"):
            raise RuntimeError("Unexpected overlay module origin: " + name)
    import importlib.metadata as metadata

    observed = {name:metadata.version(name) for name in ("transformers","huggingface_hub","tokenizers")}
    if observed != {"transformers":"4.57.1","huggingface_hub":"0.36.0","tokenizers":"0.22.2"}:
        raise RuntimeError("Unexpected import overlay version: " + repr(observed))
    provider = Path("/zimage-compat/transformers/models/siglip/modeling_siglip.py")
    if hashlib.sha256(provider.read_bytes()).hexdigest() != claim["siglip_provider_SHA256"]:
        raise RuntimeError("SigLIP provider changed before import")
    runpy.run_path("/import-run/worker.py", run_name="__main__")
    print(json.dumps({"verified_overlay_versions":observed,"siglip_provider_SHA256":claim["siglip_provider_SHA256"]}),
          flush=True)


if __name__ == "__main__":
    main()
