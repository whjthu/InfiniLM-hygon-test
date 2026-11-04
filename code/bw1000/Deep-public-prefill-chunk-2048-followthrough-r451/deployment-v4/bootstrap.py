"""Markers around the ORIGINAL actual nonserve module; no replacement validator."""
import sys
sys.stderr.write('DEEP451_PYTHON_ENTER\n');sys.stderr.flush()
import json,os,runpy,time
from pathlib import Path
expected=json.loads((Path(__file__).parent/'ORIGINAL_COMMAND.json').read_bytes())['CLI_args']
if sys.argv[1:]!=expected or '--serve'in sys.argv:raise ValueError('Frozen original nonserve arguments required')
roots=('/refactor/incoming/Deep-public-prefill-chunk-2048-followthrough-r451/source-v1/',
 '/refactor/incoming/public-Deep-native-provider-r427/source-v3/',
 '/refactor/incoming/public-GLM-native-provider-r425/',
 '/refactor/incoming/Deep-HC-fullpre-Native-r377/cpu-v1/',
 '/refactor/incoming/Deep-HC-fullpre-public-pair-r449/source-v1/')
config=expected[expected.index('--config')+1]
written=0;limit=512*1024
def marker(kind,**fields):
 global written
 raw=('DEEP451_DIAG '+json.dumps(dict(kind=kind,monotonic=time.monotonic(),**fields),ensure_ascii=True)+'\n').encode()
 if written+len(raw)<=limit:os.write(2,raw);written+=len(raw)
def observe(event,args):
 if event!='open'or not args or not isinstance(args[0],(str,bytes)):return
 path=os.fsdecode(args[0])
 if path==config or path.startswith(roots):marker('OPEN_ATTEMPT',path=path)
sys.addaudithook(observe)
sys.argv=['public_deep_model',*expected]
marker('BEFORE_ORIGINAL_RUNPY',module='public_deep_model',source_unmodified=True)
try:runpy.run_module('public_deep_model',run_name='__main__',alter_sys=False)
except BaseException as error:
 marker('ORIGINAL_MODULE_ERROR',error=repr(error));raise
else:marker('ORIGINAL_MODULE_RETURNED')
