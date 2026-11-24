"""Bind the independent zero-update audit to the live MiniCPM mask step."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.with_name("training-minicpm-checkpoint20-sdpa-terminal-audit-v1")
PREVIOUS_SHA = "cdc3cd51688b4ec41a2ba229c0288cfe949429545d13d586f587f24237efac2e"


def build():
    raw = (PREVIOUS / "read_terminal.py").read_bytes()
    if sha256(raw).hexdigest() != PREVIOUS_SHA:
        raise ValueError("previous independent terminal reader differs")
    text = raw.decode()
    for old, new, count in (
        ("cbbd25487dca4090a52c923bbfb7f60b", "3b71a062261547af8b997684603c9e6b", 1),
        ("29953", "29957", 1),
        ("training-minicpm-checkpoint20-sdpa-repeatability-target-v1", "training-minicpm-checkpoint20-native-mask-target-v1", 1),
        ("5bf1503b7af5863de7b3c8b5dc330d0e6f5bc1a8d0a20abab7988bf0cf8792b2", "0ce6733193849d6cffefe66e7034fed38a91c132375cd28f4fb98f1517150072", 1),
        ("1103606.2302", "1103606.2411", 1), ("1103606.2303", "1103606.2412", 1),
        ('("a1", "b1", "b2", "a2")', '("u1", "c1", "c2", "u2")', 1),
        ('    "read_completed.py": "2c6702af06e8edb98cae9963f0f66abcafa38c544c159efba33f727e79d5933f",',
         '''    "read_completed.py": "564c9b92d57f8e5df328f50726289d20658ce8b680126cb4d2f336a7c8f75fd9",
    "mask_compare.py": "6d370a8c0870f40d2c5164a01facaec3a785aa4a93f613e4272205846bae08a9",
    "mask_rows.py": "c1a7653985b8cb80438782b466e4c891ca080423b2d6417627bd0066e9303e47",
    "native_plugin.py": "dc120e1aac920ede6ce20767c79f95ba483015cfe59bc65ef58028ec19499460",
    "tap.py": "b191cd1eae2e4664028d61696046b573347634a33d7714c3fca3cc0becef78de",''', 1),
        ("        from route_control import comparison", "        from mask_compare import comparison", 1),
        ("sdpa-repeatability-comparison.json", "native-mask-repeatability-comparison.json", 2),
        ('    report["rejected_native_receipts"] = rejected', '''    report["rejected_native_receipts"] = rejected
    report["worker_failure"] = attempts.get("error")
    report["worker_comparison_error"] = attempts.get("comparison_error")
    report["not_attempted_arms"] = attempts.get("not_attempted_arms", [])''', 1),
    ):
        if text.count(old) != count:
            raise ValueError("previous reader anchor differs: " + old[:100])
        text = text.replace(old, new)
    ast.parse(text, feature_version=(3, 12))
    return text.encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rendered = build()
    with args.output.open("xb") as output:
        output.write(rendered)
    print(sha256(rendered).hexdigest())
