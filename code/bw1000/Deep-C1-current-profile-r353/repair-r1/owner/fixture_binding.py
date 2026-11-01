"""Reuse exact existing348 fixture; do not duplicate raw HTTP/history in source publication."""
import pathlib,hashlib
FIXTURE=pathlib.Path('/refactor/incoming/Deep-prefill-chunk-pair-r348/repair-r2/owner/frozen-toolprompt.json')
SHA='43377e6a70efdf8960989559c957241f9e725f8a2c2a55b9bf438c331275cb6d'
def fixture_path():
 assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==SHA,'Frozen actual348 fixture changed'
 return FIXTURE
