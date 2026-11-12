"""Observe one selected LoRA-B backward in the frozen SDXL replay, without repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


MODULE = "up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_q.lora_B.default"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "policy", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("backward-audit namespace already exists")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation/src"))
    from hygon_sdxl_first_step import replay, verify_reference, sha
    from gencuda.lora_backward_audit import LinearBackwardAudit
    from diffusers import UNet2DConditionModel
    import torch

    binding = verify_reference(args.reference, args.policy)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    gradient = reference["gradients_before_clip"][MODULE + ".weight"]
    args.output.mkdir()
    result = {"passed": False, "module": MODULE, "numerical_repair_applied": False,
              "original_update_gate_unchanged": True, "cross_host_raw_tensor_transfer_bytes": 0}
    try:
        with LinearBackwardAudit(UNet2DConditionModel, MODULE) as audit:
            replay(args.reference, args.weights, args.output / "replay", binding)
            result["diagnostic"] = audit.report(gradient)
            capture = args.output / "local-operands.private.pt"
            torch.save({key: value.detach().cpu() for key, value in audit.values.items()}, capture)
            result["private_capture"] = {"bytes": capture.stat().st_size, "SHA256": sha(capture)}
        result["passed"] = True
    except BaseException as error:
        result["error_type"] = type(error).__name__
        raise
    finally:
        (args.output / "backward-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
