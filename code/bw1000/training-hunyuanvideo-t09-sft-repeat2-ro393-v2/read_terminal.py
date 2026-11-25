"""Metadata consumer; never relaunch, import providers or deserialize checkpoints."""
import argparse
import json
from contract import LABEL, NODE, PARENT, REQUEST, RUN, WHOLE, load, need, save, verify_package
from review import native, post
from repeat_clock import measured_native, independent_rows, check_independent
from contract import read
from pathlib import Path
RUN = Path(RUN)


def inspect(manifest_sha, preflight_sha, epoch):
    verify_package(RUN / "src", manifest_sha)
    claim, _ = load(RUN / "src/CLAIM.json")
    need(claim.get("READ_GO") is True and claim.get("request_id") == REQUEST, "Independent read not adopted")
    _, actual_manifest = load(RUN / "src/SOURCE_FILES.json")
    need(actual_manifest["SHA256"] == manifest_sha, "Delivered consumer source manifest differs")
    _, actual_preflight = load(RUN / "preflight.json")
    need(actual_preflight["SHA256"] == preflight_sha, "Actual original preflight bytes differ")
    outcome, outcome_pin = load(RUN / "outcome.json", 256 * 1024)
    need((outcome["request_id"], outcome["parent"], outcome["node"], outcome["epoch"], outcome["deadline"])
         == (REQUEST, PARENT, NODE, epoch, epoch + WHOLE), "Original terminal scope differs")
    need(outcome["source_manifest_SHA256"] == manifest_sha, "Original terminal source binding differs")
    intent_pin = None
    if (RUN / "intent.json").exists():
        intent, intent_pin = load(RUN / "intent.json")
        need(intent["request_id"] == REQUEST and intent["epoch"] == epoch
             and intent["deadline"] == epoch + WHOLE and intent["source_manifest"]["SHA256"] == manifest_sha
             and intent["preflight"]["SHA256"] == preflight_sha, "Original intent does not bind this consumer")
    else:
        need(outcome.get("passed") is False and not outcome.get("receipts"),
             "Intent missing after actual phase invocation")
    result = {"state": "INDEPENDENT_T09_CACHED_NATIVE25_TERMINAL", "request_id": REQUEST,
              "outcome": outcome_pin, "intent": intent_pin, "planned_updates": 25,
              "passed": False, "actual_GPU_model_reruns": 0, "Model_GO": False,
              "formal17_qualified": False, "original_numerics_qualified": False,
              "recovery_qualified": False, "domain_effect_qualified": False}
    result["framework_mount_ingress"] = None
    if outcome.get("framework_mount_ingress_pin"):
        mount, mount_pin = load(RUN / "framework-mount-ingress.json", 16384)
        need(mount_pin == outcome["framework_mount_ingress_pin"] and mount == outcome["framework_mount_ingress"]
             and mount["request_id"] == REQUEST and mount["epoch"] == epoch
             and mount["step"] == outcome["receipts"]["gpu"]["row"][0]
             and mount["source_manifest_SHA256"] == manifest_sha,
             "Exact original GPU-step mounted ingress context differs")
        result["framework_mount_ingress"] = {key:mount.get(key) for key in ("passed","planned","actual_verified","phase","reason")}
        result["framework_mount_ingress_pin"] = mount_pin
    result["native"] = native(RUN / "outputs/hunyuanvideo-1.5/train" / LABEL)
    result["completed_updates"] = result["native"]["completed_updates"]
    result["missing_updates"] = 25 - result["completed_updates"]
    try:
        result["numbered_accounting"]=check_independent(independent_rows(PARENT),outcome["receipts"],
                                               PARENT,NODE,LABEL,epoch,epoch+WHOLE)
        result["post"] = post(RUN, outcome["receipts"])
        for phase, original in outcome["receipts"].items():
            actual, pin = load(RUN / (phase + "-accounting.json"))
            need(actual == original, "Original accounting receipt differs")
        result["independent_release"]=True
        result["timing"]=measured_native(RUN/"outputs/hunyuanvideo-1.5/train"/LABEL,read)
        need(result["timing"]==outcome["timing"],"T09_INDEPENDENT_TIMING_PROOF_MATCH")
        result["passed"] = outcome["passed"] is True and result["native"]["native_evidence_complete"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["readback_error"]=type(error).__name__
        if not result.get("independent_release"):
            result["post"]={"state":"UNKNOWN","error_type":type(error).__name__}
    need(len(json.dumps(result).encode()) <= 16384, "Bounded terminal summary exceeded")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch", type=float, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--preflight-sha256", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from pathlib import Path
    need(Path(args.output).parent == RUN and Path(args.output).name == "independent-terminal.json",
         "Wrong independent export destination")
    result = inspect(args.source_manifest_sha256, args.preflight_sha256, args.epoch)
    save(Path(args.output), result)
    print(json.dumps({"passed": result["passed"], "planned_updates": 25,
                      "completed_updates": result["completed_updates"], "missing_updates": result["missing_updates"]}))


if __name__ == "__main__":
    main()
