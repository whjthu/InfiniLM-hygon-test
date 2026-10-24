"""Unique terminal watcher cleanup audit; full raw consumption remains offline."""
import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
run=json.loads((r/'Kimi-static-localize-r8572-launch.json').read_text())['run']
w=json.loads((r/'Kimi-static-localize-r8572-watch.json').read_text())
out=dict(run=run,scope='STATIC_DIAGNOSIS_ONLY_NO_PERFORMANCE',owners=[],all32normal=False,
    terminal=w['terminal'],post_KFD_passed=w['post_KFD_passed'],generation_acceptance='FAILED_OBSERVED_618_NOT_RECLASSIFIED')
try:
    for node in range(4):
        p=r/'results'/f'kimi-static-localize-r8572-{run}-node{node}.json';q=json.loads(p.read_text())
        out['owners'].append(dict(node=node,completed=q['completed'],normal8=q.get('all8_owned_workers_shutdown_completed',False),
            case_labels=[v['label']for v in q.get('diagnostic_cases',[])],case_wall_s=[v['wall_s']for v in q.get('diagnostic_cases',[])],
            report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),decision=q.get('localization_decisions'),
            detail_budget_exceeded=q.get('detail_budget_exceeded'),selected_detail_complete=q.get('selected_detail_complete'),
            actual_snapshot_bytes=q.get('actual_snapshot_bytes'),error=q.get('error')))
    out['all32normal']=all(v['normal8']for v in out['owners'])
    assert out['all32normal']and all(v['completed']for v in out['owners'])and out['post_KFD_passed']
    assert out['terminal']['State']=='COMPLETED'and out['terminal']['ExitCode']=='0:0'
except BaseException as error:out['error']=repr(error);raise
finally:
    p=r/'results'/f'Kimi-static-localize-r8572-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
