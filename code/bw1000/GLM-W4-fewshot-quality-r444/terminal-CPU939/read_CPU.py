"""Stdlib-only audit of existing939 artifacts. Executes no deployment source."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCE='831609820dff256dbc64107d571bea0cb5568cbdf89320c8d099a447ac1d8fbe'
EXPECTED_RECEIPT='0ee1db3bb566f297528574ed3c4d2d2c04800800071ac7223e14344c8b0370c2'
EXPECTED_OBSERVATION='d63545ffbd38c1bf6f8dd2794445c01e2ed0c4cc867c6bdc2101236e0473a166'
EXPECTED_CASES=['actual_controller___main__','actual_launch_once___main___no_Popen',
 'actual_Run_all16_ready_and_dual_bound_latch','STOP60_then_existing_exact_TERM_KILL',
 'actual_Actors_provider_start_stop_and_primary','actual_helper_resolve_before_late_arm',
 'actual_repeated_C_watchdog_then_cancel']


def sha(raw):return hashlib.sha256(raw).hexdigest()
def audit(observation,source):
    raw=Path(observation).read_bytes();assert sha(raw)==EXPECTED_OBSERVATION
    q=json.loads(raw);assert q['accounting'][:5]==['910407.939','GLM-r444-startup-diagnostic-v4-CPU','COMPLETED','0:0','00:00:46']
    assert q['accounting'][5]=='a14r2n20'and q['GPU']==0 and not q['new_job']
    files={}
    for name,row in q['files'].items():
        b=row['rawtext'].encode();assert len(b)==row['bytes']and sha(b)==row['SHA256'],name
        files[name]=b
    assert len(files)==86
    receipt=q['receipt']['rawtext'].encode();assert receipt==files['CPU-result.json']
    assert len(receipt)==q['receipt']['bytes']and sha(receipt)==EXPECTED_RECEIPT
    r=json.loads(receipt);assert r==q['receipt_summary']and r['passed']and r['completed']and not r['errors']
    assert r['source_manifest_SHA256']==EXPECTED_SOURCE and r['actual_CPU']=='910407.939'
    assert r['GPU']==r['weights']==r['Native_compile']==0 and not r['Model']and not r['Native_dlopen']and not r['SDK_executed']
    assert all(v=='-1'for v in r['env'].values())
    assert [v['name']for v in r['cases']]==EXPECTED_CASES and all(v['passed']for v in r['cases'])
    assert not r['old_CPU918_922_926_repeated']and not r['old_quality_or_encoding_suites_repeated']
    cases={v['name']:v for v in r['cases']}
    gate=cases[EXPECTED_CASES[2]]
    assert gate['ready120']['first_all16_ready_unix']==100 and gate['ready120']['client_launch_blocked']
    assert gate['epoch2700']['client_launch_blocked']and gate['success']['startup_accepted']
    drain=cases[EXPECTED_CASES[3]]
    assert drain['normal']['normal_GPU_STOP_drain']and not drain['unknown']['normal_GPU_STOP_drain']
    assert drain['closed']['closed']and [v[-1]for v in drain['signals_recorded_not_sent']if v[0]=='pg']==[15,9]
    stack_names=[n for n in files if n.endswith('.txt')];assert len(stack_names)==1
    stack=files[stack_names[0]];assert len(stack)==2552 and stack.count(b'Timeout (0:00:20)!')==2
    assert all(v in stack for v in [b'emitter_wait',b'monitor_wait',b'watchdog_child'])
    markers=[json.loads(line)for n,b in files.items()if n.endswith('.jsonl')for line in b.decode().splitlines()]
    assert any(v['event']=='watchdog.arm.before'and v['repeat']is True and v['timeout_seconds']==20 for v in markers)
    assert any(v['event']=='watchdog.resolved'and v.get('reason')=='CPU.actual_watchdog_cancel'for v in markers)
    assert any(v['event']=='watchdog.not_armed'and v['already_resolved']for v in markers)
    assert 'libGLM_V_on_demand_Model_r403.so'not in r['proc_maps_before']+r['proc_maps_after']
    source=Path(source);mf=(source/'SOURCE_MANIFEST.json').read_bytes();assert sha(mf)==EXPECTED_SOURCE
    manifest=json.loads(mf)
    for name,h in manifest['files_SHA256'].items():assert sha((source/name).read_bytes())==h,name
    stdout=q['stdout']['rawtext'].encode();assert sha(stdout)==q['stdout']['SHA256']and len(stdout)==q['stdout']['bytes']
    return dict(passed=True,actual_CPU='910407.939',accounting=q['accounting'],CPU_wall_s=46,
        observation_SHA256=EXPECTED_OBSERVATION,receipt_SHA256=EXPECTED_RECEIPT,
        source_manifest_SHA256=EXPECTED_SOURCE,raw_files=len(files),raw_bytes=sum(map(len,files.values())),
        source_files=len(manifest['files_SHA256']),new_cases=EXPECTED_CASES,
        actual_repeated_C_watchdog_dumps=2,stack_bytes=len(stack),
        actual_process_Native_GPU_qualification=False,recorded_leaf_limits=r['recorded_leaves'],
        old918922926_repeated=False,GPU_GO=False,Model_GO=False,
        quality_completed_HTTP=0,quality_missing_original_HTTP=512,
        scope='New Python startup/keeper/diagnostic mechanism CPU only; no stall-cause or Model PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--observed',required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.observed,a.source_root)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
