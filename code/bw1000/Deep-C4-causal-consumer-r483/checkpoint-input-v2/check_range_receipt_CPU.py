"""Reject altered range declarations without accessing a checkpoint."""
from copy import deepcopy
import json
from pathlib import Path

from range_receipt import sha, validate_receipt

metadata_raw = Path(__file__).with_name("CHECKPOINT_METADATA.json").read_bytes()
meta = json.loads(metadata_raw)
fixture = dict(schema=1, layer=meta["layer"], total_selected_weight_bytes=13_144_320,
               original_config_SHA256=meta["original_config_SHA256"],
               original_index_SHA256=meta["original_index_SHA256"],
               header_SHA256={name: row["header_SHA256"] for name, row in meta["headers"].items()},
               whole_shard_hashed=False, Native_dlopen=False, Runtime=False, GPU=False,
               weight_values_exported=False,
               weights={name: dict(bytes=row["bytes"], SHA256=sha(name.encode()))
                        for name, row in meta["weights"].items()})
raw = json.dumps(fixture).encode()
assert validate_receipt(raw, sha(raw), metadata_raw) == fixture
name = next(iter(fixture["weights"]))
changed = []
for value in ({"layer": True}, {"Native_dlopen": 0}, {"GPU": True},
              {"total_selected_weight_bytes": 13_144_321}, {"header_SHA256": {}},
              {"original_index_SHA256": "0" * 64}):
    changed.append(fixture | value)
for update in ({"SHA256": "z" * 64}, {"bytes": 0}):
    wrong_range = deepcopy(fixture)
    wrong_range["weights"][name].update(update)
    changed.append(wrong_range)
missing = deepcopy(fixture)
del missing["weights"][name]
changed.append(missing)
bad_receipts = [json.dumps(item).encode() for item in changed]
bad_receipts.extend((b'{"schema":1,' + raw[1:], raw + bytes(16 * 1024)))
for data in bad_receipts:
    try:
        validate_receipt(data, sha(data), metadata_raw)
    except ValueError:
        continue
    raise AssertionError("bad range receipt accepted")
for receipt_sha, metadata in (("0" * 64, metadata_raw), (sha(raw), metadata_raw + b" ")):
    try:
        validate_receipt(raw, receipt_sha, metadata)
    except ValueError:
        continue
    raise AssertionError("mismatched outer pin accepted")
print(json.dumps(dict(valid_fixture_ranges=7, rejected=len(bad_receipts) + 2,
                      actual_original_receipt_consumed=False, checkpoint_payload_bytes_read=0)))
