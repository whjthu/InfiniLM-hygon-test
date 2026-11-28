"""Verify native Euler wire identity up to the single proven order-only field."""

from hashlib import sha256
from itertools import permutations
import json

FIELD = "_use_default_values"
ORIGINAL_WIRE = "b9e0375a791fc3c9666c8dee997ee0e5e29d624a00920139daf4880ea154c46f"
ORIGINAL_NORMALIZED = "db68ddadf96c8a48e78d7b497f5b54b5f8fa3b9659c3b80ef87e0df85730d802"
QUALIFICATION_SHA256 = "9cec4af5add8b6a3dd781f43e69d7a07b4527602519d0f743e16e0e5037dc936"
CONFIG_SOURCE = "4a7af9be48913edfa77f3d32c375c997fa15b53c3f04efd5c371b36c5c6c1960"
EULER_SOURCE = "ecee4a110e09b91edf76715434afa60a1b34281c2e08166114f7be0a1facfa63"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def wire(config):
    return (json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def raw_state(config):
    need(type(config) is dict and 1 <= len(config) <= 128 and all(type(key) is str for key in config),
         "BOUNDED_EULER_CONFIG_MAPPING")
    raw = wire(config)
    need(0 < len(raw) <= 16 << 10, "BOUNDED_EFFECTIVE_EULER_WIRE")
    return digest(raw)


def establish(config, historical):
    need(historical == ORIGINAL_WIRE, "EXACT_RETAINED_2762_EULER_WIRE")
    current = raw_state(config)
    values = config.get(FIELD)
    need(type(values) is list and len(values) == 7
         and all(type(item) is str and 0 < len(item) <= 128 for item in values) and len(set(values)) == 7,
         "EXACT_SEVEN_DISTINCT_EULER_DEFAULT_MEMBERS")
    members = sorted(values)
    normalized = raw_state(dict(config, **{FIELD: members}))
    need(normalized == ORIGINAL_NORMALIZED, "EXACT_2803_NORMALIZED_FULL_EULER_CONFIG")
    for index, order in enumerate(permutations(members), 1):
        rebuilt = raw_state(dict(config, **{FIELD: list(order)}))
        if rebuilt == historical:
            return dict(current_raw_SHA256=current, normalized_SHA256=normalized,
                        reconstructed_2762_SHA256=rebuilt, permutation_index=index, permutation_bound=5040,
                        qualification_worker_SHA256=QUALIFICATION_SHA256)
    raise ValueError("ORIGINAL_EULER_WIRE_NOT_RECONSTRUCTED")


def verify(facts, rows, historical, *, complete):
    artifact = facts["scheduler_config_bridge"]
    need(type(artifact) is dict and set(artifact) == {"effective_config", "receipt"},
         "EXACT_SCHEDULER_BRIDGE_INPUT_AND_RECEIPT")
    computed = establish(artifact["effective_config"], historical)
    need(computed == artifact["receipt"]
         and computed["current_raw_SHA256"] == facts["effective_scheduler_config_SHA256"],
         "EXACT_RECOMPUTED_EULER_WIRE_BRIDGE")
    raw = computed["current_raw_SHA256"]
    successful = [row for row in rows if row["outcome"] == "completed"]
    need(all(row.get("scheduler_raw_before_SHA256") == row.get("scheduler_raw_after_SHA256") == raw
             for row in successful), "EXACT_PER_SLOT_RAW_EULER_WIRE_STABILITY")
    if complete:
        need(len(successful) == 6 and facts.get("scheduler_final_raw_SHA256") == raw,
             "EXACT_FINAL_RAW_EULER_WIRE_STABILITY")
    return computed
