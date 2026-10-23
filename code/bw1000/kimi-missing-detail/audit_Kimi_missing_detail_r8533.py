"""Terminal cleanup audit followed by CPU consumption; never inside Engine."""
import pathlib,json,hashlib
from kimi_missing_detail_offline_r8533 import consume
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
run=json.loads((r/'Kimi-missing-detail-r8533-launch.json').read_text())['run']
w=json.loads((r/'Kimi-missing-detail-r8533-watch.json').read_text())
out=dict(run=run,scope='CAPTURE_DIAGNOSIS_ONLY',generation_acceptance='FAILED_OBSERVED',
    model_qualified=False,performance_qualified=False,owners=[],all32normal=False,
    terminal=w['terminal'],post_KFD_passed=w['post_KFD_passed'],offline_complete=False)
try:
    for node in range(4):
        p=r/'results'/('kimi-missing-detail-r8533-'+run+'-node'+str(node)+'.json');q=json.loads(p.read_text());c=q['native_capture']
        assert q['completed']and c['status']=='CAPTURE_COMPLETE_OFFLINE_PENDING'and len(c['cases'])==2
        assert all(len(v)==4 for v in c['cases'])
        out['owners'].append(dict(node=node,normal8=q.get('all8_owned_workers_shutdown_completed',False),report_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),prefix_IDs_equal=[all(v['all_prefix_IDs_equal']for v in case)for case in c['cases']]))
    out['all32normal']=all(v['normal8']for v in out['owners'])
    assert out['all32normal']and out['post_KFD_passed']
    assert out['terminal']['State']=='COMPLETED'and out['terminal']['ExitCode']=='0:0'
    result=consume(r,run)
    p=r/'results'/('Kimi-missing-detail-r8533-offline-'+run+'.json');assert not p.exists();p.write_text(json.dumps(result,indent=2))
    out.update(offline_complete=True,offline_path=str(p),offline_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),
        Oconfirm_crosssession_native_differences=len(result['Oconfirm_vs_previous_Oa_differences']),
        O_vs_W_native_differences=len(result['O_vs_W_native_target_differences']))
except BaseException as exc:
    out['error']=repr(exc);raise
finally:
    p=r/'results'/('Kimi-missing-detail-r8533-audit-'+run+'.json');assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
