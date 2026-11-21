"""Recompute SF3D head MAE from the private, pinned material records."""

import math
from hashlib import sha256
from pathlib import Path


MATERIAL_FIELDS = {"albedo MAE": ("albedo", 3), "roughness MAE": ("roughness", 1),
                   "metallic MAE": ("metallic", 1)}
POLICY = "sf3d/native-material-head/sample-equal-mae-v1"
RGBA_ARTIFACT = "SF3D source RGBA"
QUERY_ARTIFACT = "SF3D source query positions"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def scorer_sha256():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def vector(record, sample_id, rgba, queries, checkpoint, query_count):
    require(record["sample_id"] == sample_id and record["status"] == "scored"
            and record.get("RGBA_SHA256") == rgba and record.get("query_SHA256") == queries
            and "checkpoint_SHA256" in record
            and record["checkpoint_SHA256"] == checkpoint, "SF3D material record identity differs")
    fields = record["material"]
    require(isinstance(fields, dict) and fields.keys() == {field for field, _ in MATERIAL_FIELDS.values()},
            "SF3D material head fields differ")
    for field, channels in MATERIAL_FIELDS.values():
        spatial = field == "albedo" and checkpoint is not None
        rows = fields[field] if spatial else [fields[field]]
        require(isinstance(rows, list) and len(rows) == (query_count if spatial else 1)
                and all(isinstance(values, list) and len(values) == channels
                        and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                                for value in values) for values in rows),
                "SF3D material head values or query denominator differ")
    return fields


def verify_material_mae(row, identity, plan, inventory, checkpoints, verify_artifacts):
    require(callable(verify_artifacts) and identity["case"] == "T11.material-head"
            and identity.get("scorer_SHA256") == scorer_sha256() and plan.get("material_policy") == POLICY,
            "missing SF3D material MAE scorer, policy or readable records")
    ids = plan["sample_ids"]
    labels = plan["material_references"]
    images = plan["material_RGBA_artifacts"]
    queries = plan["material_query_artifacts"]
    query_count = plan["query_points_per_sample"]
    require(type(query_count) is int and 0 < query_count <= 16_384,
            "SF3D material query denominator differs")
    require([item["sample_id"] for item in images] == ids and all(item["status"] == "scored" for item in images),
            "SF3D material RGBA source inventory differs")
    require([item["sample_id"] for item in queries] == ids and all(item["status"] == "scored" for item in queries),
            "SF3D material query source inventory differs")
    verify_artifacts(images, RGBA_ARTIFACT)
    require(verify_artifacts(queries, QUERY_ARTIFACT) == dict.fromkeys(ids, query_count),
            "SF3D query count differs from the frozen plan")
    rgba = {item["sample_id"]: item["artifact"]["SHA256"] for item in images}
    positions = {item["sample_id"]: item["artifact"]["SHA256"] for item in queries}
    require([item["sample_id"] for item in labels] == ids and all(item["status"] == "scored" for item in labels),
            "SF3D material references must cover the frozen plan")
    metric = identity["metric"]
    field, channels = MATERIAL_FIELDS[metric]
    truth = verify_artifacts(labels, metric)
    refs = {name: vector(truth[name], name, rgba[name], positions[name], None, query_count)[field] for name in ids}
    for arm, samples in inventory["arms"].items():
        require(row[arm + "_failed"] == 0 and row[arm + "_scored"] == len(ids),
                "SF3D formal material MAE requires all planned predictions")
        predictions = verify_artifacts(samples, metric)
        errors = []
        for name in ids:
            observed = vector(predictions[name], name, rgba[name], positions[name], checkpoints[arm], query_count)[field]
            values = observed if field == "albedo" else [observed]
            errors.append(math.fsum(abs(value - ref) for sample in values
                                    for value, ref in zip(sample, refs[name], strict=True)) / len(values))
        recomputed = math.fsum(errors) / (channels * len(ids))
        require(math.isclose(row[arm], recomputed, rel_tol=0, abs_tol=1e-12),
                "SF3D material MAE differs from the verified head outputs")
    return (identity.get("dataset_SHA256"), identity.get("scorer_SHA256"),
            ids, query_count, images, queries, labels, inventory["arms"])
