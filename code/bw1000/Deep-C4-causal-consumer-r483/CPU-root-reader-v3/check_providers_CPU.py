"""Exercise independent CPU-v2 DSO hashing against local regular files."""
import hashlib
import json
from pathlib import Path
import tempfile

from read_original_build import hash_regular_provider, verify_project_providers

plan_path = Path(__file__).resolve().parents[1] / "CPU-v2" / "BUILD_PLAN.json"
plan_bytes = plan_path.read_bytes()
assert hashlib.sha256(plan_bytes).hexdigest() == "587a51c72b7bfb3342676a90b73ffa7cab84bdbfd9c8271ff46d23048fea5082"
providers = json.loads(plan_bytes)["dynamic_SHA256"]
fixture_pins = dict(providers)
project = sorted(path for path in providers if path.startswith("/refactor/"))
reads = []
with tempfile.TemporaryDirectory() as directory:
    files = {}
    for ordinal, path in enumerate(project):
        file = Path(directory) / str(ordinal)
        data = path.encode() * 10000
        file.write_bytes(data)
        files[path] = file
        fixture_pins[path] = hashlib.sha256(data).hexdigest()

    def observed(path):
        reads.append(path)
        return hash_regular_provider(files[path])

    result = verify_project_providers(fixture_pins, observed)
    assert result == {path: fixture_pins[path] for path in project}
    assert reads == project
    for path in project:
        file = files[path]
        data = file.read_bytes()
        file.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        try:
            verify_project_providers(fixture_pins, observed)
        except ValueError as error:
            assert str(error).endswith(path)
        else:
            raise AssertionError(f"accepted changed provider bytes: {path}")
        file.write_bytes(data)

print(json.dumps(dict(frozen_plan_provider_count=len(providers), project_fixture_count=len(project),
                      changed_same_size_rejected=len(project), actual_target_DSO_reads=0)))
