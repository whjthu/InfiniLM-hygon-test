"""Execute the actual tiny stdin bootstrap locally; no SSH or endpoint work."""
import json
from pathlib import Path
import runpy
import shlex
import struct
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
ns = runpy.run_path(str(D/'relay_once.py'))
bootstrap, frame = ns['BOOTSTRAP'], ns['frame']
local = [sys.executable, '-I', '-S', '-B', '-c', bootstrap]
script = b"import sys\nassert sys.stdin.buffer.read()==b'COMPRESSED_FIXTURE'\nprint('EXACT_SUFFIX_OK')\n"
result = subprocess.run(local, input=frame(script, b'COMPRESSED_FIXTURE'), capture_output=True, timeout=5)
assert result.returncode == 0 and result.stdout == b'EXACT_SUFFIX_OK\r\n' and not result.stderr
rejected = []
for name, payload in [('short_prefix', b'\x01'), ('short_script', struct.pack('<Q', 20)+b'abc'),
                      ('oversized_script_prefix', struct.pack('<Q', 131073))]:
    result = subprocess.run(local, input=payload, capture_output=True, timeout=5)
    assert result.returncode != 0 and not result.stdout
    rejected.append(name)
try:
    ns['bounded_call'](['x'*30000], b'', 1, time.time()+300, time.monotonic()+300, D, 'not-run')
except ValueError as error:
    assert str(error) == 'SSH argv exceeds Windows command-line bound'
    rejected.append('oversized_argv_before_Popen')
else:
    raise AssertionError('oversized argv accepted')
for interpreter in ['python3', '/usr/local/python3.12/bin/python3']:
    argv = ['ssh', '-T', '-F', 'EXISTING_CONFIG_NOT_OPENED', 'EXISTING_HOST_NOT_CONTACTED',
            interpreter+' -I -S -B -c '+shlex.quote(bootstrap)]
    assert len(subprocess.list2cmdline(argv)) < 30000
print(json.dumps(dict(passed=True, actual_bootstrap_child=True, exact_suffix_preserved=True,
                     rejected=rejected, remote_calls=0, Torch_or_SDK_imports=0)))
