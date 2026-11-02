"""Regression for the actual723 state shape/dtype width; no Native/GPU."""
from pathlib import Path
import importlib.util,contextlib,io,copy,json
import numpy as np
D=Path(__file__).resolve().parent;CPU=D.parent/'cpu-v1'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
with contextlib.redirect_stdout(io.StringIO()):fixtures=load('fixture390state',CPU/'check_consumer_CPU.py')
fixed=load('fixed390state',D/'consumer_statebytes.py');original=fixtures.consumer;oracle=fixtures.oracle
first=fixtures.fixtures(8575);second=fixtures.fixtures(8576,first)
for past,records in [(8575,first),(8576,second)]:
 for record in records:
  for stage in ['before','after_sdk','after_paged','post_commit']:
   for k in [8,9,11]:
    bits=np.full((8,512),0 if past==8575 else 0x3f80,np.uint16)
    if stage=='post_commit'and k==11:bits.fill(0x3f80 if past==8575 else 0x4000)
    record[stage][k]=dict(shape=[8,512],dtype=9,raw=bits.view(np.uint8).reshape(-1).copy())
assert fixed.validate_same_cache(first,oracle)['passed']
try:original.validate_same_cache(second,oracle,first,committed=False)
except ValueError as e:assert 'cannot reshape'in str(e)
else:raise AssertionError('Original realwidth state bug not reproduced')
assert fixed.validate_same_cache(second,oracle,first,committed=False)['passed']
checks=['original_bug_reproduced_8x512_BF16_8192bytes','corrected_shape_dtype_rawbytes_carry_pass']
for case in ['dtype','shape','bytes']:
 bad=copy.deepcopy(second)
 for stage in ['before','after_sdk','after_paged']:
  for k in [8,9,11]:
   f=bad[0][stage][k]
   if case=='dtype':f['dtype']=8
   elif case=='shape':f['shape']=[4,1024]
   else:f['raw'][0]^=1
 try:fixed.validate_same_cache(bad,oracle,first,committed=False)
 except AssertionError:pass
 else:raise AssertionError('State carry mismatch accepted: '+case)
 checks.append('reject_nextstate_'+case)
print(json.dumps(dict(passed=True,checks=checks,actual_saved_state_metadata=dict(shape=[8,512],dtype_enum=9,bytes=8192),Native_dlopen=False,GPU=False,new_compile=False)))
