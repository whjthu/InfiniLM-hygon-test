"""Verify the frozen 72-sample domain workload at the native training boundary."""

import json

from scope import DOMAIN, DOMAIN_DATA, need, pin

BEFORE = ("input_ids", "labels", "pixel_values")
AFTER = ("attention_mask", "labels", "logits_to_keep")
PREPARE_SHA = "28fe9cb927390ad423add8b98411a90fb2d1c801ead124b354802bdd0779ecf6"


def signatures(rows, roster):
    need(len(rows) == len(roster) == 72, "Original domain denominator differs")
    result = {}
    for ordinal, (row, item) in enumerate(zip(rows, roster), 1):
        need(row["ordinal"] == item["ordinal"] == ordinal
             and row["source_row_SHA256"] == item["row_SHA256"]
             and row["source_ID_SHA256"] == item["source_ID_SHA256"]
             and row["label_and_logit_mask_native_semantics_checked"] is True
             and row["finite_CPU_input_fields"] is True, "Original domain input evidence differs")
        key = tuple(row["before_fields"][name]["SHA256"] for name in BEFORE)
        need(key not in result, "Original domain sample signature is ambiguous")
        result[key] = {"ordinal": ordinal, "row_SHA256": item["row_SHA256"],
                       "after": {name: row["after_fields"][name]["SHA256"] for name in AFTER}}
    return result


def verified_domain():
    contract = json.loads(pin(DOMAIN / "source/DATA_CONTRACT.json", extent=2234,
        expected="1c018bda3c45cad41c83cbbdec1c389f659b411b55eaeb8562d5256886733909"))
    need(contract["target_data"] == str(DOMAIN_DATA) and len(contract["files"]) == 14,
         "Original domain data contract differs")
    for name, value in contract["files"].items():
        pin(DOMAIN_DATA / name, expected=value["SHA256"], extent=value["bytes"])
    roster = json.loads(pin(DOMAIN / "source/INPUT_ROWS.json", extent=26356,
        expected="ffa9205ca05440d3d8cb9ef50a3c259623d1e3a892d830c538f3a9b014e61696"))["rows"]
    rows = json.loads(pin(DOMAIN / "native-caller-rows.json", extent=167302,
        expected="c21d4dbeff9ae4c9dad1a6b0eeb56146969fe13cde916a7bae0cf218be98ee61"))
    return signatures(rows, roster)


def before_sample(inputs, domain, projection):
    key = tuple(projection(inputs[name])["SHA256"] for name in BEFORE)
    need(key in domain, "Actual prepared collator sample was absent from original domain")
    return domain[key]


def after_sample(inputs, member, fingerprint):
    need(member is not None and all(fingerprint(inputs[name])["SHA256"] == member["after"][name]
                                   for name in AFTER), "Native forward labels or masks differ from the collator")
    return {"ordinal": member["ordinal"], "row_SHA256": member["row_SHA256"]}


def verify_updates(before1, after1, before2, after2):
    need(before1["parameters"] == after1["parameters"] == before2["parameters"]
         and before2["parameters"] != after2["parameters"]
         and before1["optimizer"] != after1["optimizer"]
         and before2["optimizer"] != after2["optimizer"],
         "First warmup step and second parameter update differ")
    need(all(lr == 0 for lr in before1["learning_rates"])
         and bool(before1["learning_rates"])
         and bool(before2["learning_rates"])
         and all(lr > 0 for lr in before2["learning_rates"]),
         "Actual learning-rate warmup differs")
