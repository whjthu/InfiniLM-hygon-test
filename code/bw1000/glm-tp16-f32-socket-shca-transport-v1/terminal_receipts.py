"""Read exact JSON receipts after a caller has verified terminal accounting."""
import hashlib
import json
from pathlib import Path
import time


def collect_receipts(root, names, deadline, *, interval=0.5, max_attempts=40,
                     monotonic=time.monotonic, sleep=time.sleep, max_bytes=131072):
    """Retry absent files only. Preserve partial results and surface malformed receipts."""
    names = tuple(names)
    if not names or len(set(names)) != len(names) or interval <= 0 or max_attempts < 1 or max_bytes < 1:
        raise ValueError("invalid receipt collection policy")
    for name in names:
        path = Path(name)
        if path.is_absolute() or path.drive or ".." in path.parts or not path.parts:
            raise ValueError("receipt path must be relative to the original run")
    pending = dict.fromkeys(names)
    receipts = {}
    observations = []
    while pending and len(observations) < max_attempts and monotonic() < deadline:
        newly_visible = []
        for name in tuple(pending):
            if monotonic() >= deadline:
                break
            try:
                with (root / name).open("rb") as stream:
                    raw = stream.read(max_bytes + 1)
            except FileNotFoundError:
                continue
            if len(raw) > max_bytes:
                raise ValueError(f"oversized receipt: {name}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"receipt must be a JSON object: {name}")
            receipts[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), value=value)
            newly_visible.append(name)
            del pending[name]
        observations.append(dict(newly_visible=newly_visible, missing_count=len(pending)))
        remaining = deadline - monotonic()
        if pending and remaining > 0 and len(observations) < max_attempts:
            sleep(min(interval, remaining))
    return dict(receipts=receipts, missing=list(pending), observations=observations)
