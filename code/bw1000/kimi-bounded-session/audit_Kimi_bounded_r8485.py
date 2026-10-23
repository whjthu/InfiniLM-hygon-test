"""CPU-only independent terminal audit; never launches GPU, cleanup or Engine."""
import pathlib,json,hashlib,datetime
from kimi_matched_trace_r8467 import validate_records,digest
from online_service_metrics_r8455 import evaluate,distribution
from kimi_bounded_batch_wait_r8485 import consume_plan
DSO='9001bf2bfa7e0821d52d9bec61ed5d9bf028c93380673fa880af3ed52e9c7388'
def resolve_evidence_path(root,value,run):
    root=pathlib.Path(root);p=pathlib.PurePosixPath(value)
    assert str(p).startswith('/refactor/results/') and '..'not in p.parts
    out=root.joinpath(*p.relative_to('/refactor').parts)
    assert run in out.name and out.name.startswith('Kimi-bounded-arrival-r8485-')
    return out

def audit(root):
    root=pathlib.Path(root);issues=[];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    intent=json.loads((root/'Kimi-bounded-r8485-launch.json').read_text());run=intent['run']
    watch=json.loads((root/'Kimi-bounded-r8485-watch.json').read_text())
    manifest=root/'incoming/Kimi-bounded-manifest-r8485.json';m=json.loads(manifest.read_text())
    assert sha(manifest)==intent['manifest_SHA256']
    for name,h in {**m['source_sha256'],**m['existing_dependency_sha256']}.items():assert sha(root/'incoming'/name)==h
    reports=[];report_SHA={};epoch_results=[]
    for n in range(4):
        p=root/'results'/f'kimi-bounded-r8485-{run}-node{n}.json'
        q=json.loads(p.read_text());reports.append(q);report_SHA[str(n)]=sha(p)
        assert q['completed'] and q['all8_owned_workers_shutdown_completed'] and q['historical_B16M4_anchor_passed'] and q['before8QA_passed']
        assert len(q['QA'])==16 and len({(v['epoch'],v['task'])for v in q['QA']})==16
        assert q['library_sha256']==DSO and q['bounded_wait_four_epochs_qualified']
        assert list(q['bounded_wait_qualification'])==['O1','W1','W2','O2']
        assert all(v['qualified']for v in q['bounded_wait_qualification'].values())
    q=reports[0];epochs=q['actual_arrival_epochs'];assert [e['label']for e in epochs]==['bounded-O1','bounded-W1','bounded-W2','bounded-O2']
    assert [e['wait_s']for e in epochs]==[0,.01,.01,0]
    for epoch in epochs:
        label=epoch['label'].removeprefix('bounded-')
        jp=resolve_evidence_path(root,epoch['capture_path'],run);ep=resolve_evidence_path(root,epoch['event_path'],run)
        assert sha(jp)==epoch['capture_SHA256'] and sha(ep)==epoch['event_SHA256']
        jobs=[json.loads(s)for s in jp.read_text().splitlines()];events=[json.loads(s)for s in ep.read_text().splitlines()]
        checks=q['bounded_wait_oracle_checks'][label];assert len(checks)==len(jobs)==q['bounded_wait_qualification'][label]['oracle_jobs']
        records=[]
        for actual,check in zip(jobs,checks):
            assert actual['job']==check['job'] and actual['job_SHA256']==check['job_SHA256']==digest(actual['job'])
            oracle=check['reference_oracle'];assert actual['actual_ids']==check['live_actual_ids']==oracle['cpu_argmax_ids']
            assert oracle['finite'] and oracle['native_equals_CPU'] and oracle['native_ids']==oracle['cpu_argmax_ids']
            records.append(dict(job=actual['job'],job_SHA256=actual['job_SHA256'],reference_ids=oracle['cpu_argmax_ids'],oracle=oracle))
        header=dict(scope='MATCHED_SHAPE_JOBTRACE_ONLY',rows=32,reference_flags=67,candidate_flags=71,eos_token_ids=[163585,163586],output_limits={str(i):128 for i in range(8)})
        outputs=validate_records(header,records)
        assert [outputs[k]for k in range(8)]==epoch['actual_outputs']
        prompts={k:p for j in jobs if j['job']['kind']=='prefill'for k,p in zip(j['job']['keys'],j['job']['inputs'])}
        inputs=[prompts[k]for k in range(8)]
        plan,plan_SHA=consume_plan(root/'incoming/Kimi-fixed-arrival-plan-r8485.json',inputs)
        assert plan_SHA==q['bounded_wait_plan_file_SHA256'] and digest(plan)==epoch['arrival_plan_SHA256']
        assert digest(inputs)==q['bounded_wait_native_identity']['inputs_SHA256']
        for k in range(8):assert [e['token']for e in events if e['kind']=='token'and e.get('request')==k]==outputs[k]
        schedule=epoch['producer_schedule'];assert [v['request']for v in schedule]==list(range(8))
        base=schedule[0]['target_t']
        for i,v in enumerate(schedule):
            assert abs(v['target_t']-base-plan['offsets_s'][i])<1e-8
            assert v['send_t']>=v['target_t'] and abs(v['lateness_s']-(v['send_t']-v['target_t']))<1e-9
            assert next(e['arrival_t']for e in events if e['kind']=='arrival'and e['request']==i)==v['actual_ingress_t']
        merged=sorted(events,key=lambda e:e['t']);arr=[e.get('arrival_t',e['t'])for e in merged if e['kind']=='arrival']
        metric=evaluate(dict(events=merged,window=[min(arr),max(e['t']for e in merged)],cards=32,boundary='emitter'))
        assert all(metric[k]==epoch['metric'][k]for k in metric)
        assert metric['statuses']==dict(completed=8,unfinished=0,cancelled=0,rejected=0,failed=0)
        epoch_results.append(dict(label=label,qualified=True,jobs=len(jobs),metric=metric,
             published_rate_name='service-window useful post-first output rate; arrival-to-drain denominator includes prefill, NOT puredecode',
             source_SHA256=dict(jobs=sha(jp),events=sha(ep)),sender_lateness_s=distribution([v['lateness_s']for v in schedule]),
             actual_ingress_offsets_s=[v['actual_ingress_t']-schedule[0]['actual_ingress_t']for v in schedule],
             oracle_phase_s=q['bounded_wait_qualification'][label]['oracle_phase_s']))
    assert watch['terminal']['State']=='COMPLETED' and watch['terminal']['ExitCode']=='0:0' and watch['post_KFD_passed']
    assert len(watch['post_KFD'])==4 and all(not v['kfd_pids']for v in watch['post_KFD'])
    return dict(revision=8485,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),qualified=True,issues=[],run=run,
                scope='Finite coordinator-observed O/W/W/O, each actual trace independently CPUargmax checked. SameTick newO is paired control; old588 diagnostic only. No clientdelivery/infinite-service/puredecode claim.',
                epochs=epoch_results,terminal=watch['terminal'],post_KFD_passed=True,report_SHA256=report_SHA)
if __name__=='__main__':
    root=pathlib.Path.home()/'hygon-inference-refactor-20261014';run=json.loads((root/'Kimi-bounded-r8485-launch.json').read_text())['run']
    try:out=audit(root)
    except BaseException as exc:out=dict(revision=8485,qualified=False,issues=[repr(exc)],run=run)
    p=root/'results'/f'Kimi-bounded-r8485-audit-{run}.json'
    with p.open('x')as f:json.dump(out,f,indent=2)
    print(json.dumps(dict(path=str(p),qualified=out['qualified'],issues=out['issues'])))
