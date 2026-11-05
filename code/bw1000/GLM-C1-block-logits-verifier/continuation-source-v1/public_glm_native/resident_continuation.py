"""Run the frozen diagnostic inside the original GLM ResidentContext."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

from .owner_compare import OWNERS
from .accepted_prefix_continuation import compare_continuation
from .freeze_continuation_inputs import freeze, PARENT as PARENT_PLAN_SHA256

PLAN_SHA256 = "f71463189b3da010aabac90cdb28a14a0a232e58e22cad05dfa60b4c321ecac4"
MEMORY_FLOOR = 512 * 1024**2


def _apply_original_policies(lib, ctx, rank, check):
    check(lib.modern_glm_vector_policy_r8540(rank, 0))
    for name in ("owned_prefix_policy_r1669", "feature_bypass_r1406", "indexer_policy_r998"):
        check(getattr(lib, "modern_glm_model_" + name)(ctx, 0))
    check(lib.modern_glm_model_decode_leases_policy_r1356(ctx, 1))
    transient_report = lib.modern_glm_model_transient_report_r5688
    transient_report.argtypes = [ctypes.c_void_p]
    transient_report.restype = ctypes.c_char_p
    transient = json.loads(transient_report(ctx))
    if (transient["rows"], transient["capacity"], transient["enabled"],
            transient["GPU_logical_bytes"]) != (1, 32768, False, 0):
        raise RuntimeError("unexpected GLM transient or cache geometry")

    batch_policy = lib.modern_glm_model_batch_cache_policy_r3068
    batch_policy.argtypes = [ctypes.c_void_p, ctypes.c_int]
    check(batch_policy(ctx, 1))
    for name, value in (("glm_reduced_output_policy_r9032", 0),
                        ("glm_attention_cohort_policy_r9054", 1),
                        ("glm_ragged_policy_r9136", 1)):
        setter = getattr(lib, name)
        setter.argtypes = [ctypes.c_int, ctypes.c_int]
        check(setter(rank, value))


def run_resident_prefix_continuation(resident_context, *, stream, plan_path, run,
                                     configure_native_environment, verified_native_sha):
    """Borrow the original context, gates and run failure ledger.

    The enclosing worker owns Slurm and deadline enforcement. The returned
    report follows original lifecycle completion, never an early READY.
    """
    stage, rank = resident_context.stage, resident_context.rank

    def publish_failure(error):
        try:
            run.fail(error, stage=stage, rank=rank)
        except Exception as publication_error:
            error.add_note(f"GLM comparison failure publication: {publication_error!r}")

    try:
        if (run.config.get("prefix_continuation_execution_authorized") is not True
                or verified_native_sha != run.config["Native_SHA256"]
                or len(verified_native_sha) != 64
                or any(c not in "0123456789abcdef" for c in verified_native_sha)
                or verified_native_sha == "9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b"):
            raise ValueError("continuation requires a verified new Native build and claim")
        raw = freeze(Path(plan_path).read_bytes())
        if hashlib.sha256(raw).hexdigest() != PLAN_SHA256:
            raise ValueError("unexpected frozen private input bytes")
        cases = json.loads(raw)["cases"]
        if [c["geometry"] for c in cases] != ["below", "above", "crossing_prompt_slice"]:
            raise ValueError("unexpected frozen input order")
        run.deadline_check("before GLM block comparison context", 120)
    except BaseException as error:
        publish_failure(error)
        raise

    torch = resident_context.torch
    min_free = None
    reports = []
    try:
        with resident_context as resident:
            try:
                if not resident.report["all16_loaded_and_postload_4GiB_passed"]:
                    raise RuntimeError("original postload floor has not passed")
                configure_native_environment(os.environ)
                lib, ctx = resident.binding.lib, resident.ctx
                check, ptr = resident.binding.check, resident.binding.ptr
                _apply_original_policies(lib, ctx, rank, check)

                def control(value):
                    # Keep the original post-forward 512 MiB floor and make a
                    # local violation visible to every owner in the same vote.
                    nonlocal min_free
                    run.deadline_check("GLM block comparison control")
                    free, _ = torch.cuda.mem_get_info(rank)
                    min_free = free if min_free is None else min(min_free, free)
                    floor = value.new_tensor([int(free >= MEMORY_FLOOR)])
                    frame = torch.cat((value, floor))
                    check(lib.modern_glm_model_control(ctx, frame.numel(), ptr(frame), stream()))
                    if frame[-1].item() != OWNERS:
                        raise RuntimeError("GLM comparison breached the original 512 MiB floor")
                    value.copy_(frame[:-1])

                for case in cases:
                    run.deadline_check("before GLM block comparison case")
                    geometry = case["geometry"]
                    result = compare_continuation(
                        lib=lib, ctx=ctx, torch=torch, stage=stage, rank=rank,
                        control=control, ptr=ptr, stream=stream, prompt=case["prompt"],
                        chunk_widths=case["chunk_widths"], proposal=case["proposal"],
                        sequence=case["sequence"], accepted=case["accepted"],
                        continuation=case["continuation"],
                    )
                    reports.append(dict(geometry=geometry, **result))

                check(lib.glm_ragged_policy_r9136(rank, 0))
                check(lib.glm_attention_cohort_policy_r9054(rank, 0))
                check(lib.modern_glm_vector_policy_r8540(rank, 0))
            except BaseException as error:
                publish_failure(error)
                raise
        lifecycle = json.loads(Path(resident_context.lifecycle_path).read_bytes())
        expected = dict(stage=stage, rank=rank, completed=True, normal=True,
                        destroy_attempts=1, shutdown_attempts=1,
                        resident_reset_RC=0, destroy_RC=0, shutdown_RC=0)
        if any(lifecycle.get(key) != value for key, value in expected.items()):
            raise RuntimeError("original owner lifecycle did not close normally")
        result = {
            "stage": stage, "rank": rank, "Native_SHA256": verified_native_sha,
            "private_plan_SHA256": PLAN_SHA256, "parent_private_plan_SHA256": PARENT_PLAN_SHA256,
            "minimum_driver_free_bytes": min_free,
            "normal_shutdown": True, "cases": reports,
        }
        run.write(f"prefix-continuation-{stage}-{rank}.json", result)
        return result
    except BaseException as error:
        publish_failure(error)
        raise
