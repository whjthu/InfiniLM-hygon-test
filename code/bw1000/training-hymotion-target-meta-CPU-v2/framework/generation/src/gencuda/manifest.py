from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .source_identity import config_digest, source_identity
from .trellis_output_contract import DIAGNOSTIC_MULTIBOX, LATENT_ONLY, validate_output_contract


def sha256_path(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _declared_execution_identity(
    config: Mapping[str, Any], environ: Mapping[str, str], config_key: str, env_key: str
) -> str | None:
    configured = config.get(config_key)
    environment = environ.get(env_key)
    for source, value in ((config_key, configured), (env_key, environment)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{source} must be a non-empty string when declared")
    if configured is not None and environment is not None and configured != environment:
        raise ValueError(f"conflicting {config_key} and {env_key}")
    return configured if configured is not None else environment


def new_manifest(
    command: str,
    model: str,
    config: dict[str, Any],
    runtime: dict[str, Any],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "model": model,
        "config": config,
        "runtime": runtime,
        "host": platform.node(),
        "pid": os.getpid(),
        "artifacts": [],
        "metrics": {},
        "stage_timings": {},
        "checks": {
            "finite_tensors": None,
            "nonzero_finite_gradients": None,
            "checkpoint_resume": None,
            "current_adapter": None,
            "main_forward_device": None,
            "reward_variance_nonzero": None,
            "log_prob_alignment": None,
            "turbo_8step_regression": None,
        },
        "observations": {},
        "versions": {
            "container": config.get("container", config.get("container_image")),
            "model_revision": config.get("model_revision"),
            "source_shas": config.get("source_shas", {}),
        },
        "logs": [],
        "provenance": {
            "executed_source": source_identity(),
            "config_sha256": config_digest(config),
            "source_sha": config.get("source_sha"),
            "source_shas": config.get("source_shas", {}),
            "model_sha": config.get("model_sha"),
            "dataset_sha": config.get("dataset_sha"),
            "seed": config.get("seed"),
            "cards": config.get("cards"),
            "precision": config.get("precision", "bfloat16"),
            "dataset_manifest": config.get("dataset_manifest"),
            "adapter_sha": config.get("adapter_sha"),
            "cpu_decoder_source_identity": None,
            "global_step": None,
            "measured_steps": None,
        },
        "execution": {
            "status": "created",
            "host_label": _declared_execution_identity(
                config, environment, "host_label", "GENCUDA_HOST_LABEL"
            ),
            "run_id": _declared_execution_identity(
                config, environment, "run_id", "GENCUDA_RUN_ID"
            ),
            "checkpoint": None,
            "resume_from": config.get("resume_from"),
        },
    }


def _nonnegative_int(metrics: Mapping[str, Any], key: str) -> int | None:
    if key not in metrics:
        return None
    value = metrics[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"metrics.{key} must be a non-negative integer")
    return value


def _validate_file_entries(entries: Any, field: str, *, required: bool) -> list[dict[str, str]]:
    if not isinstance(entries, list):
        raise ValueError(f"backend result {field} must be a list")
    if required and not entries:
        raise ValueError("backend result must contain at least one artifact")
    validated: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"backend result {field}[{index}] must be a mapping")
        path_value = entry.get("path")
        declared_sha = entry.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"backend result {field}[{index}].path must be a non-empty string")
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            raise ValueError(f"backend result {field}[{index}].sha256 must be a SHA-256 digest")
        if any(character not in "0123456789abcdef" for character in declared_sha.lower()):
            raise ValueError(f"backend result {field}[{index}].sha256 must be a SHA-256 digest")
        path = Path(path_value)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"backend result {field}[{index}] is missing or empty: {path}")
        actual_sha = sha256_path(path)
        if actual_sha != declared_sha.lower():
            raise ValueError(f"backend result {field}[{index}] SHA-256 mismatch: {path}")
        validated.append({"path": str(path), "sha256": actual_sha})
    return validated


def mark_failed(manifest: dict[str, Any], stage: str, error: BaseException) -> None:
    manifest["status"] = "failed"
    execution = manifest.setdefault("execution", {})
    execution["status"] = "failed"
    execution["stage"] = stage
    execution["error"] = {"type": type(error).__name__, "message": str(error)}


def _validate_trellis_output_capability(
    capability: Any, config: Mapping[str, Any], artifacts: list[dict[str, str]]
) -> dict[str, Any]:
    if not isinstance(capability, Mapping):
        raise ValueError("TRELLIS backend result output_capability must be a mapping")
    expected_keys = {
        "schema", "requested", "produced", "full_mesh_available", "formal_output_contract",
        "primary_artifact", "shape_decoder_executed", "texture_decoder_executed",
        "diagnostic_preview", "formal_mesh_available", "textured_glb_available",
        "decoded_fields_artifact",
    }
    if set(capability) != expected_keys:
        raise ValueError("TRELLIS output_capability fields mismatch")
    configured = validate_output_contract(config)
    requested = capability["requested"]
    if requested != configured:
        raise ValueError("TRELLIS output_capability requested contract does not match config")
    expected_produced = (
        "textured_glb" if requested == "full_mesh" else
        ("latent_with_diagnostic_multibox" if requested == DIAGNOSTIC_MULTIBOX else LATENT_ONLY)
    )
    if capability["schema"] != "trellis-output-capability-v1":
        raise ValueError("TRELLIS output_capability schema is unsupported")
    if capability["produced"] != expected_produced:
        raise ValueError("TRELLIS output_capability produced value is invalid")
    for name in (
        "full_mesh_available", "formal_output_contract", "shape_decoder_executed",
        "texture_decoder_executed", "diagnostic_preview", "formal_mesh_available",
        "textured_glb_available",
    ):
        if type(capability[name]) is not bool:
            raise ValueError(f"TRELLIS output_capability.{name} must be a boolean")
    if any(capability[name] != (requested == "full_mesh") for name in (
        "full_mesh_available", "formal_mesh_available", "textured_glb_available", "formal_output_contract"
    )):
        raise ValueError("TRELLIS output_capability formal mesh flags differ from requested contract")
    decode_cpu_fields = config.get("decode_cpu_fields", False)
    if type(decode_cpu_fields) is not bool:
        raise ValueError("TRELLIS config decode_cpu_fields must be a boolean")
    if decode_cpu_fields and requested not in {LATENT_ONLY, "full_mesh"}:
        raise ValueError("TRELLIS CPU decoder capability requires latent_only or full_mesh")
    if capability["shape_decoder_executed"] is not decode_cpu_fields:
        raise ValueError("TRELLIS shape decoder capability differs from config")
    if capability["texture_decoder_executed"] is not decode_cpu_fields:
        raise ValueError("TRELLIS texture decoder capability differs from config")
    if capability["diagnostic_preview"] != (requested == DIAGNOSTIC_MULTIBOX):
        raise ValueError("TRELLIS output_capability diagnostic_preview is invalid")
    primary = capability["primary_artifact"]
    if not isinstance(primary, Mapping) or set(primary) != {"path", "sha256", "role"}:
        raise ValueError("TRELLIS output_capability primary_artifact fields mismatch")
    if primary["role"] != ("textured_glb" if requested == "full_mesh" else "latent"):
        raise ValueError("TRELLIS output_capability primary_artifact role differs")
    if not isinstance(primary["path"], str) or not isinstance(primary["sha256"], str):
        raise ValueError("TRELLIS output_capability primary_artifact is invalid")
    if not any(
        artifact["path"] == primary["path"] and artifact["sha256"] == primary["sha256"]
        for artifact in artifacts
    ):
        raise ValueError("TRELLIS output_capability primary_artifact is not a verified artifact")
    decoded = capability["decoded_fields_artifact"]
    if not decode_cpu_fields:
        if decoded is not None:
            raise ValueError("TRELLIS decoded fields artifact requires CPU decoder opt-in")
    else:
        if not isinstance(decoded, Mapping) or set(decoded) != {"path", "sha256", "role"}:
            raise ValueError("TRELLIS decoded fields artifact fields mismatch")
        if decoded["role"] != "decoded_fields_and_geometry":
            raise ValueError("TRELLIS decoded fields artifact role is invalid")
        if not any(
            artifact["path"] == decoded["path"] and artifact["sha256"] == decoded["sha256"]
            for artifact in artifacts
        ):
            raise ValueError("TRELLIS decoded fields artifact is not verified")
    return dict(capability)


def merge_backend_result(manifest: dict[str, Any], result: Mapping[str, Any]) -> None:
    required_keys = ("artifacts", "logs", "metrics", "stage_timings", "checks")
    missing_keys = [key for key in required_keys if key not in result]
    if missing_keys:
        raise ValueError(f"backend result is missing {missing_keys[0]}")
    artifacts = _validate_file_entries(result["artifacts"], "artifacts", required=True)
    logs = _validate_file_entries(result["logs"], "logs", required=False)
    metrics = result.get("metrics")
    stage_timings = result["stage_timings"]
    checks = result.get("checks")
    observations = result.get("observations", {})
    if not isinstance(stage_timings, Mapping):
        raise ValueError("backend result stage_timings must be a mapping")
    if not isinstance(metrics, Mapping) or not isinstance(checks, Mapping):
        raise ValueError("backend result must contain mapping metrics and checks")
    if not isinstance(observations, Mapping):
        raise ValueError("backend result observations must be a mapping")
    measured_steps = _nonnegative_int(metrics, "steps")
    explicit_global_step = _nonnegative_int(metrics, "global_optimizer_step")
    output_capability = None
    if manifest.get("command") == "infer" and manifest.get("model") == "trellis-2-4b":
        if "output_capability" not in result:
            raise ValueError("TRELLIS backend result is missing output_capability")
        output_capability = _validate_trellis_output_capability(
            result["output_capability"], manifest.get("config", {}), artifacts
        )
        full_mesh = output_capability["requested"] == "full_mesh"
        sparse_support = result.get("sparse_support")
        if full_mesh or sparse_support is not None:
            from .trellis_sparse_support import validate_proof

            manifest["provenance"]["sparse_support"] = validate_proof(
                sparse_support, artifacts, full_mesh=full_mesh,
                max_sparse_tokens=manifest.get("config", {}).get("max_sparse_tokens", 1024),
            )
        if checks.get("formal_output_contract") is not full_mesh:
            raise ValueError("TRELLIS checks.formal_output_contract differs from requested contract")
        decode_cpu_fields = manifest.get("config", {}).get("decode_cpu_fields", False)
        cpu_identity = result.get("cpu_decoder_source_identity")
        if decode_cpu_fields:
            if not isinstance(cpu_identity, Mapping) or set(cpu_identity) != {
                "clean", "revision", "source_sha256"
            }:
                raise ValueError("TRELLIS CPU decoder source identity is missing")
            if cpu_identity["clean"] is not True or cpu_identity["revision"] != (
                "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
            ):
                raise ValueError("TRELLIS CPU decoder source identity is invalid")
            source_hashes = cpu_identity["source_sha256"]
            if (
                not isinstance(source_hashes, Mapping)
                or set(source_hashes) != {"decoder", "geometry"}
                or any(not isinstance(value, str) or len(value) != 64 for value in source_hashes.values())
            ):
                raise ValueError("TRELLIS CPU decoder source hashes are invalid")
            manifest["provenance"]["cpu_decoder_source_identity"] = dict(cpu_identity)
        elif cpu_identity is not None:
            raise ValueError("TRELLIS CPU decoder source identity requires opt-in")
        export_report = result.get("cpu_texture_export_report")
        if full_mesh:
            from .trellis_cpu_texture_export import validate_export
            if not isinstance(export_report, Mapping) or set(export_report) != {"path", "sha256"}:
                raise ValueError("TRELLIS full_mesh requires export report")
            if dict(export_report) not in artifacts:
                raise ValueError("TRELLIS export report is not a verified artifact")
            report = validate_export(
                Path(export_report["path"]), export_report["sha256"], manifest["config"]["cpu_texture_export"],
                Path(output_capability["primary_artifact"]["path"]),
                Path(output_capability["decoded_fields_artifact"]["path"]),
            )
            for entry in (report["decode_report"], report["correspondence"]):
                if entry not in artifacts:
                    raise ValueError("TRELLIS texture correspondence/report is not a verified artifact")
            if report["decoder_source"] != {"revision": cpu_identity["revision"], "files": cpu_identity["source_sha256"]}:
                raise ValueError("TRELLIS texture/decoder source identity differs")
            decode_path = Path(report["decode_report"]["path"])
            if decode_path.stat().st_size > 4 * 1024 ** 2:
                raise ValueError("TRELLIS decode report exceeds metadata budget")
            decode_input = json.loads(decode_path.read_text(encoding="utf-8")).get("input")
            latent_entry = sparse_support["latent"]
            if (
                not isinstance(decode_input, Mapping) or set(decode_input) != {"path", "sha256"}
                or Path(decode_input["path"]).resolve() != Path(latent_entry["path"]).resolve()
                or decode_input["sha256"] != latent_entry["sha256"]
                or sha256_path(decode_path) != report["decode_report"]["sha256"]
            ):
                raise ValueError("TRELLIS decoded geometry is not bound to native-support latent")
            manifest["provenance"]["cpu_texture_export_report"] = dict(export_report)
        elif export_report is not None:
            raise ValueError("TRELLIS texture report requires full_mesh")

    manifest["artifacts"] = artifacts
    manifest["logs"] = logs
    manifest["stage_timings"] = dict(stage_timings)
    manifest["metrics"] = dict(metrics)
    manifest["checks"].update(checks)
    manifest["observations"] = dict(observations)
    if output_capability is not None:
        manifest["output_capability"] = output_capability
    adapter_sha = metrics.get("adapter_sha_after")
    if adapter_sha is not None:
        manifest["provenance"]["adapter_sha"] = adapter_sha
    manifest["provenance"]["measured_steps"] = measured_steps
    if explicit_global_step is not None:
        manifest["provenance"]["global_step"] = explicit_global_step
    manifest["status"] = "completed"
    manifest["execution"]["status"] = "completed"
    manifest["execution"]["stage"] = "completed"


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
