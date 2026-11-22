"""Pure metadata gates for a new target-created Gemma domain initialization."""
import math


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def clock(origin, deadline, now, maximum=2400):
    need(all(type(x) in (int, float) and math.isfinite(x) for x in (origin, deadline, now)),
         "nonfinite or untyped original clock")
    need(0 < origin <= now < deadline <= origin + maximum, "outside original epoch")
    return deadline - now


def fresh_config(config):
    need(config.get("initialization_mode") == "TARGET_FRESH_DOMAIN_ONLY"
         and config.get("seed") == config.get("data_seed") == 42
         and "initial_trainables" not in config,
         "fresh domain initialization must not consume a reference initial snapshot")


def cohort(records):
    need(len(records) == 8 and [r["rank"] for r in records] == list(range(8)), "eight ordered rank records required")
    for r in records:
        f = r["named"]
        need(f.get("finite") is True and f.get("tensors") == 476
             and f.get("elements") == 29802496 and isinstance(f.get("SHA256"), str)
             and len(f["SHA256"]) == 64, "incomplete actual fresh trainable fingerprint")
    need(len({r["named"]["SHA256"] for r in records}) == 1,
         "actual fresh initial trainables differ across ranks")
    return records[0]["named"]["SHA256"]
