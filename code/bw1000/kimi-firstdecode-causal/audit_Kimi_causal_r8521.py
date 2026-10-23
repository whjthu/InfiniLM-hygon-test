"""One terminal CPU evidence audit; no new GPU/audit retry or service claim."""
import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
d=json.loads((r/'Kimi-causal-r8521-launch.json').read_text());run=d['run']
m=json.loads((r/'incoming/Kimi-causal-manifest-r8521.json').read_text())
def path(value):
    value=str(value)
    if value=='/refactor':return r
    if value.startswith('/refactor/'):return r/value[len('/refactor/'):]
    return pathlib.Path(value)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in m['source_sha256'].items():assert sha(r/'incoming'/name)==h,name
watch=json.loads((r/'Kimi-causal-r8521-watch.json').read_text())
out=dict(run=run,scope='Finite diagnostic cleanup/evidence only',
    generation_acceptance='FAILED_OBSERVED historical crosspolicy finaltokens',
    performance_qualified=False,native_diagnosis_complete=False,
    all32normal=False,terminal=watch.get('terminal'),post_KFD_passed=watch.get('post_KFD_passed',False),owners=[])
try:
    for node in range(4):
        p=r/'results'/f'kimi-causal-r8521-{run}-node{node}.json'
        q=json.loads(p.read_text());causal=q.get('native_causal',{})
        for phase in causal.get('phases',[]):
            assert len(phase)==4
            for entry in phase:
                v=json.loads(path(entry['outputs_path']).read_text())
                assert v and all(x['firstdiff']is None for x in v)
        out['owners'].append(dict(node=node,report_SHA256=sha(p),completed=q['completed'],
            normal8=q.get('all8_owned_workers_shutdown_completed',False),
            QA_count=len(q.get('QA',[])),historical_anchor=q.get('historical_B16M4_anchor_passed',False),
            causal_status=causal.get('status'),first_stage=causal.get('first_differing_stage_observed'),
            first_layer=causal.get('first_differing_layer_observed'),error=q.get('error')))
    out['all32normal']=all(v['normal8']for v in out['owners'])
    out['native_diagnosis_complete']=all(v['completed']and v['QA_count']==16 and v['historical_anchor']and v['causal_status']=='FINITE_DIAGNOSIS_COMPLETE'for v in out['owners'])
    assert out['terminal']and out['terminal']['State']=='COMPLETED'and out['terminal']['ExitCode']=='0:0'
    assert out['all32normal']and out['native_diagnosis_complete']and out['post_KFD_passed']
except BaseException as exc:
    out['error']=repr(exc)
    raise
finally:
    p=r/'results'/f'Kimi-causal-r8521-audit-{run}.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
