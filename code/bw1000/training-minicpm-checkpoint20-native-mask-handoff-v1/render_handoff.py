"""Bind the already qualified standby handoff to the current mask scope and occupant."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRIOR = HERE.with_name("training-minicpm-checkpoint20-sdpa-handoff-v1")
OLD_SHA = "4cdf7d18e9b3298b18b846b24905781bf50c9136f9addb8dc8704fc7dea5c728"
REQUEST = "3b71a062261547af8b997684603c9e6b"
TARGET_SHA = "0ce6733193849d6cffefe66e7034fed38a91c132375cd28f4fb98f1517150072"
CURRENT_STEP, CURRENT_STANDBY = "1103606.2308", "9213447e542641c7a220b5fca3015f46"
TARGET_PUB = "6ef7d5bac284a8297fbe9c662e28f4b784638fa3"
EXPECTED_SHA = "0fce7423fe4ee3c61c2607844b566a6c90340de2c4bdda61ec0281c499452657"
WORKER_SHA = "1853714bc3124cdecf9a9b9b8d2a588581dbb5bf80e82e973920c98e2c3286fa"
INSTALL_SHA = "c9c52e2ea0e70f0f602840ce11ac2405dc1e45ef551738d62fb5cd12ddc9c39d"
ASSETS_SHA = "7f2cbf864c1264ea037fc95c2fec47cf6a3c8f4d5bf5ee7371fb08645c7f13d6"


def replacement(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError("frozen handoff anchor differs: " + old[:100])
    return raw.replace(old, new)


def build():
    old_path = PRIOR / "handoff.py"
    raw = old_path.read_bytes()
    if sha256(raw).hexdigest() != OLD_SHA:
        raise ValueError("qualified handoff source differs")
    source = raw.decode()
    for old, new in (
        ("cbbd25487dca4090a52c923bbfb7f60b", REQUEST),
        ("1103606.2208", CURRENT_STEP),
        ("2e5a9fcb35134860a93091403b06f8a3", CURRENT_STANDBY),
        ("training80-dcu-standby-2e5a9fcb3513-f13r1n19", "training80-dcu-standby-9213447e5426-f13r1n19"),
        ("training-minicpm-checkpoint20-sdpa-repeatability-target-v1", "training-minicpm-checkpoint20-native-mask-target-v1"),
        ("training-minicpm-checkpoint20-sdpa-repeatability-source-cbbd25487dca", "training-minicpm-checkpoint20-native-mask-source-3b71a0622615"),
        ("training-minicpm-sdpa-handoff-", "training-minicpm-native-mask-handoff-"),
        ("2c6784036e865dbbe40adff4264c3221bb8462e2", TARGET_PUB),
        ("5bf1503b7af5863de7b3c8b5dc330d0e6f5bc1a8d0a20abab7988bf0cf8792b2", TARGET_SHA),
        ("7bdf34de2195fd9d8f7bdb279f0bc6fa1a7ddcf426f1890f61d3f52ad514a02e", EXPECTED_SHA),
        ('worker = read(STANDBY / "worker.json")', f'worker = read(STANDBY / "worker.json", {WORKER_SHA!r})'),
        ('    manifest = read(TARGET / "SOURCE_FILES.json", TARGET_SHA)', f'''    staged = read(SOURCE / "source-install.json", {INSTALL_SHA!r})
    assets = read(SOURCE / "asset-preflight-receipt.json", {ASSETS_SHA!r})
    require(staged['request'] == assets['request'] == REQUEST
            and staged['publication'] == assets['publication'] == TARGET_PUBLICATION
            and staged['manifest_SHA256'] == assets['preflight']['source_manifest_SHA256'] == TARGET_SHA
            and staged['installed_exact_files'] == assets['installed_exact_files'] == 58
            and assets['rc'] == 0 and assets['preflight']['original_asset_gate'] == 'PASSED'
            and assets['preflight']['actual_GPU_steps'] == 0, 'original mask asset receipt differs')
    manifest = read(TARGET / "SOURCE_FILES.json", TARGET_SHA)'''),
        ('        require(terminal() is None, "running predecessor has a terminal marker")',
         '''        require(first.get("live_standby_recent_cards") == 8, "ordinary standby lacks eight recent compute counters")
        require(terminal() is None, "running predecessor has a terminal marker")'''),
        ('        return dict(parent=parent, original_diagnostic_unstarted=unused,\n'
         '                    accounting=original[0], active_node_steps=steps)',
         '''        recent_cards = None
        if original[0][2] == "RUNNING" and original[0][0] in steps and steps[STANDBY_STEP]["state"] == "RUNNING":
            cards = [(pci, read(STANDBY / f"card-{index}.json")) for index, pci in enumerate(worker["pci"])]
            observed = time.time()
            recent_cards = sum(row["pci"] == pci and row["synthetic"] is True and row["dtype"] == "BF16"
                               and row["active"] is True and row["ready"] is True and row["calls"] > 0
                               and 0 <= observed - row["epoch"] < 25 for pci, row in cards)
        return dict(parent=parent, original_diagnostic_unstarted=unused,
                    accounting=original[0], active_node_steps=steps, live_standby_recent_cards=recent_cards)'''),
    ):
        source = replacement(source, old, new)
    ast.parse(source, feature_version=(3, 12))
    return source.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = build()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(request=REQUEST, previous_step=CURRENT_STEP, bytes=len(raw), SHA256=sha256(raw).hexdigest())))


if __name__ == "__main__":
    main()
