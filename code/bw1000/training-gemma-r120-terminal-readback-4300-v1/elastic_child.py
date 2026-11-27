import json,os,sys
from pathlib import Path
assert os.environ['RANK']=='0' and os.environ['WORLD_SIZE']=='1'
with (Path(sys.argv[1])/'elastic-child.json').open('x') as s:json.dump(dict(local_rank=0,native_torchelastic=True,model_invoked=False),s)
print('native TorchElastic log redirect fixture')
