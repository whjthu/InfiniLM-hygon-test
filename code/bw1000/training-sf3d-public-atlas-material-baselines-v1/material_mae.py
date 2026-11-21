"""Score uniform UV atlas material vectors over the frozen object denominator."""

import math


FIELDS = {"albedo": 3, "roughness": 1, "metallic": 1}


def vector(values):
    if not isinstance(values, dict) or values.keys() != FIELDS.keys():
        raise ValueError("expected all three material fields")

    def channels(name):
        width = FIELDS[name]
        field = values[name]
        if not isinstance(field, (list, tuple)) or len(field) != width:
            raise ValueError("incorrect channel count: " + name)
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in field):
            raise ValueError("invalid normalized material value: " + name)
        return tuple(field)

    return {name: channels(name) for name in FIELDS}


def score(planned, reference, predictions):
    if (not planned or len(set(planned)) != len(planned)
            or set(planned) != reference.keys() or not predictions.keys() <= reference.keys()):
        raise ValueError("predictions or reference differ from the planned object IDs")
    reference = {name: vector(reference[name]) for name in planned}
    per_object = {}
    failures = {}
    for name in planned:
        if name not in predictions:
            failures[name] = "missing_prediction"
            continue
        try:
            prediction = vector(predictions[name])
        except ValueError:
            failures[name] = "invalid_material_vector"
            continue
        per_object[name] = {
            field: math.fsum(abs(a - b) for a, b in zip(prediction[field], target)) / FIELDS[field]
            for field, target in reference[name].items()
        }
    complete = not failures
    metrics = {field: math.fsum(row[field] for row in per_object.values()) / len(planned)
               for field in FIELDS} if complete else None
    return dict(planned=len(planned), scored=len(per_object), failed=len(failures), failures=failures,
                complete=complete, per_object=per_object, MAE=metrics)
