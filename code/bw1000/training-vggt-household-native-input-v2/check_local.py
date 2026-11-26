"""Offline gate checks. No target subprocesses or asset files are opened."""

import ast
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch


here = Path(__file__).resolve().parent
root = here.parents[2]
v2 = runpy.run_path(str(here/'carrier.py'))
old = runpy.run_path(str(root/'code/bw1000/training-vggt-household-native-input-v1/carrier.py'))
scheduler = runpy.run_path(str(root/'code/bw1000/training-vggt-3493-cp25-cpu-reload-v1-host-carrier.py'))
assert len(v2['REQUEST']) == len(v2['READ_REQUEST']) == len(v2['CHECK_REQUEST']) == 32
token = v2['command']()
assert all(token.count(x) == 1 for x in ('--gres=none','-c4','--mem=49152M','00:05:00','--exact','worker'))
assert '-I' in token and '-S' in token and v2['OLD_LABEL'] not in str(token)
assert token[-2].startswith('/public/') and '\\' not in token[-2]
assert old['LABEL'] == v2['OLD_LABEL'] and old['INPUT_SHA'] == v2['INPUT_PIN'] and old['SOURCE_SHA'] == v2['SOURCE_PIN']
parsed = ast.parse((here/'carrier.py').read_text('utf-8'))
assert sum(isinstance(x,ast.Call) and getattr(x.func,'attr',None) == 'Popen' for x in ast.walk(parsed)) == 1


def rejects(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError('must reject')


row = ['1103606.4999',v2['LABEL'],'COMPLETED','0:0','300',v2['NODE'],'cpu=4,mem=48G,node=1',
       '2026-11-27T00:00:00','2026-11-27T00:05:00']
intent = {'epoch':1795737600.0,'deadline':1795737925.0}
assert v2['terminal']([row],intent,scheduler) == row
for index,value in ((0,v2['OLD_ROW'][0]),(1,v2['OLD_LABEL']),(2,'RUNNING'),(4,'301'),(5,'other'),
                    (6,'cpu=4,gres/dcu=1,mem=48G,node=1'),(8,'2026-11-27T00:06:00')):
    changed = row.copy()
    changed[index] = value
    rejects(v2['terminal'],[changed],intent,scheduler)
rejects(v2['terminal'],[row,row],intent,scheduler)
cancelled = [*row[:2],'CANCELLED by 0','0:9','0',*row[5:8],row[7]]
assert v2['terminal']([cancelled],intent,scheduler) == cancelled

rows = [{'slot':i, 'state':'NATIVE_CPU_QUALIFIED','native_input_SHA256':'b'*64} for i in range(1,21)]
rows[14] = {'slot':15,'state':'FROZEN_UNAVAILABLE'}
result = {'numeric_step':row[0],'input_manifest_SHA256':v2['INPUT_PIN'],'backend_SHA256':v2['BACKEND_PIN'],
          'planned_common_pairs':20,'planned_scoring_slots':40,'frozen_unavailable':1,'ready_for_scoring':False,
          'actual_GPU':0,'actual_predictions':0,'model_weights_opened':0,'rows':rows,
          'qualified_input_pairs':19,'failed_eligible':0,'passed':True,'ordered_native_batch_SHA256':'c'*64}


def stream(value):
    return b'ordinary runtime output\n' + v2['PREFIX'] + v2['pack'](value)


assert v2['native_result'](stream(result),row[0]) == result
rejects(v2['native_result'],stream(result)*2,row[0])
rejects(v2['native_result'],b'',row[0])
rejects(v2['native_result'],stream(result),v2['OLD_ROW'][0])
rejects(v2['native_result'],stream(dict(result,rows=rows[:-1])),row[0])
rejects(v2['native_result'],stream(dict(result,input_manifest_SHA256='f'*64)),row[0])
rows[0] = {'slot':1,'state':'NATIVE_CPU_FAILED'}
failure = dict(result,qualified_input_pairs=18,failed_eligible=1,passed=False,ordered_native_batch_SHA256=None)
assert v2['native_result'](stream(failure),row[0])['failed_eligible'] == 1
assert len(rows) == 20
rows[0] = {'slot':1,'state':'NATIVE_CPU_QUALIFIED','native_input_SHA256':'b'*64}


class Child:
    returncode = 0

    def poll(self):
        return self.returncode

    def communicate(self, timeout):
        assert 0 < timeout <= 310
        return stream(result), b''


def fake_write(path, data):
    with path.open('xb') as handle:
        handle.write(data)
    return v2['pin'](data)


current = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
actual_fixture = [*row[:4],'0',*row[5:7],current,current]
mock_scheduler = dict(scheduler,accounting=lambda:[actual_fixture],lock=nullcontext,fresh=lambda:None)
worker_namespace = {'write':fake_write,'sync':lambda path:None}
now = time.time()
publication = 'e'*40
program = (here/'carrier.py').read_bytes()
data = runpy.run_path(str(here/'prepare.py'))['assemble']('launch',program,v2,publication,{
    'request':v2['REQUEST'],'publication':publication,'parent':v2['PARENT'],'node':v2['NODE'],
    'sole_zero_GRES_authority':True,'ordinary_conflict_excluded':True,'custody_evidence_SHA256':'d'*64,
    'issued_epoch':now,'expires_epoch':now+400})
live = v2['launch'].__globals__
with TemporaryDirectory() as temp:
    with patch.dict(live,{'RUN':Path(temp)/'unique','admissions':lambda mode:mock_scheduler,
                         'source_api':lambda:worker_namespace}), patch('subprocess.Popen',return_value=Child()) as started:
        launched = v2['launch'](data)
        assert launched['passed'] and launched['qualified_input_pairs'] == 19 and started.call_count == 1
        assert '--gres=none' in started.call_args.args[0]
        try:
            v2['launch'](data)
        except FileExistsError:
            assert started.call_count == 1
        else:
            raise AssertionError('already committed RUN must not launch')
        rejected = dict(data,authority=dict(data['authority'],request=v2['OLD_REQUEST']))
        rejects(v2['launch'],rejected)
        assert started.call_count == 1
    check_data = Path(temp)/'CHECK_DATA.json'
    prepared = subprocess.run([sys.executable,'-I','-S','-B','-X','utf8',str(here/'prepare.py'),
                               'check',str(check_data)],cwd=temp,capture_output=True,check=True)
    assert check_data.read_bytes() == v2['pack']({'mode':'check','request':v2['CHECK_REQUEST']})
    output = json.loads(prepared.stdout)
    assert output['program'] == v2['pin'](program) and output['argv'] == token
print(json.dumps({'pass':True,'asset_bytes_opened':0,'remote_calls':0,'captured_srun_tokens':len(token),
                  'planned':20,'fixture_native_verified':19,'frozen_missing_ordinal':15,
                  'fixture_partial_native_verified':18,'process_adapter_calls':1,'actual_native_runs':0,
                  'canonical_prepare_from_external_cwd':True},sort_keys=True))
