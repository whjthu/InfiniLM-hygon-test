"""Set full frozen loader environment before the target Python interpreter starts."""
from pathlib import Path
import hashlib,json,os,sys
W=Path('/run-scope');b=json.loads((W/'src/LIBRARY_BINDING.json').read_bytes());p=Path('/qualified-preflight/evidence/loader-environment.private.json');raw=p.read_bytes()
assert len(raw)==b['loader_environment_pin']['bytes'] and hashlib.sha256(raw).hexdigest()==b['loader_environment_pin']['SHA256']
frozen=json.loads(raw);env=os.environ.copy();env['LD_LIBRARY_PATH']=frozen['LD_LIBRARY_PATH'];env['PATH']=frozen['PATH']
assert len(sys.argv)==2 and sys.argv[1] in ['worker.py']
os.execvpe(sys.executable,[sys.executable,'-B','-u',str(W/'src'/sys.argv[1])],env)
