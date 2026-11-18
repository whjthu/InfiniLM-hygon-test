"""One tiny owned local child; no SSH or original-artifact IO."""
import json
import subprocess
import sys
import time
from run_once import reap_held

child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time;time.sleep(30)'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
result = reap_held(child, time.time()+2)
assert result['reaped'] and result['kill_requested'] and child.poll() is not None
print(json.dumps(dict(owned_child_reaped=True, observations=result, SSH=False, original_IO=False)))
