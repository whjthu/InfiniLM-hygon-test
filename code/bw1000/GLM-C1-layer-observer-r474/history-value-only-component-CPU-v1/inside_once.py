"""Queue delay consumes the same original inside epoch; no fresh timeout."""
import json
import math
import os
from pathlib import Path
import time

D = Path(__file__).resolve().parent
intent = json.loads((D.parent/'intent.json').read_bytes())
assert intent['inside_deadline'] == intent['epoch'] + 270
assert intent['work_deadline'] <= intent['epoch'] + 265
seconds = min(intent['inside_seconds'], intent['inside_deadline'] - time.time(),
              intent['deadline'] - time.time() - 60)
assert math.isfinite(seconds) and seconds > 2, 'original inside epoch exhausted in queue'
os.execv('/usr/bin/timeout', ['/usr/bin/timeout', '--kill-after=10s', str(int(seconds)),
         '/usr/bin/python3', '-B', '-u', str(D/'build_once.py')])
