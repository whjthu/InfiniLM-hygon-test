"""One private C2M1 residency, four stage handles, bounded whole-set fail-stop."""
import argparse,datetime,json,os,socket,time,traceback
from pathlib import Path
from backend import Backend,clean_env
from contract import validate,bundle,sha,STAGES,TERMINAL,NATIVE,GPU_SECONDS,CANDIDATE
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent


def verify(root,directory,g):
    validate(g);bundle(directory,g['controller_manifest_SHA256'])
    bundle(Path(root)/'incoming/GLM-prefill-window-interleave-r361/runtime-r1/owner',g['owner_manifest_SHA256'])
    cp=Path(root)/'candidates/GLM-prefill-window-interleave-r361/runtime-r1/CPU-result.json'
    assert sha(cp)==g['CPU_receipt_SHA256'];q=json.loads(cp.read_bytes())
    assert q['completed']and q['passed']and q['GPU']==0 and q['Native_dlopen']is False
    assert q['owner_manifest_SHA256']==g['owner_manifest_SHA256']and q['controller_manifest_SHA256']==g['controller_manifest_SHA256']
    base=bundle(Path(root)/'incoming/GLM-two-cohort-Model-r355/owner','32f24ae36f4148c9eac9f86294a4bcccb3c1f269d6b1123cb9750d88a0693366')
    current=json.loads((Path(root)/'incoming/GLM-prefill-window-interleave-r361/runtime-r1/owner/SOURCE_MANIFEST.json').read_bytes())
    assert current['base_355_owner_manifest_SHA256']=='32f24ae36f4148c9eac9f86294a4bcccb3c1f269d6b1123cb9750d88a0693366' and current['interleave_snapshot_manifest_SHA256']=='11cf0416e344e0fe8795c809c2cf54a87588f9dc6c8093a8d203749068deed8e'
    assert current['files_SHA256']['prefill_interleave.py']=='54861403adad4a6143eb260a5735fb0fdfbaf9e5202457202dc4a4a7054bb932'
    assert set(current['files_SHA256'])==set(base['files_SHA256'])|{'prefill_interleave.py'}
    for name,h in base['files_SHA256'].items():
        if name not in current['changed_from_355']:assert current['files_SHA256'][name]==h
    from namespace_binding import verify_namespace
    verify_namespace(root,directory)


def terminal(o):return o['row']is not None and o['row'][2].split()[0].rstrip('+')in TERMINAL
def success(o):return terminal(o)and o['row'][2]=='COMPLETED'and o['row'][3]=='0:0'and o['srun_exit']==0


def visible_json(path,seconds=90):
    until=time.monotonic()+seconds
    while True:
        try:
            q=json.loads(path.read_bytes())
            if q.get('completed')is True:return q
        except(FileNotFoundError,json.JSONDecodeError):pass
        if time.monotonic()>=until:raise TimeoutError('Original completed artifact visibility unresolved')
        time.sleep(1)


def collect(root,backend,g):
    owners=[];signatures=[];logs=[];tail=None;bounds=[0,18,38,58,78]
    for stage in range(4):
        step=backend.actual[stage];log=Path(root)/'logs'/(backend.label(stage)+'-'+step+'.log')
        text=log.read_text(errors='replace')
        assert text.count('GLM_R361_NORMAL '+str(stage))==1 and'Using network IBext_v8'in text and'Using network Socket'not in text
        logs.append(dict(stage=stage,actual_step=step,SHA256=sha(log),IBext_v8=True,normal8=True))
        for rank in range(8):
            p=Path(root)/'results'/('GLM-r361-'+g['run']+'-stage%d-rank%d.json'%(stage,rank));q=visible_json(p)
            assert q['worker_shutdown_completed']and q['post_close_Torch_liveness']and q['normal_reset_close']and q['normal_shutdown']
            assert q['actual_producer']==step and q['stage']==stage and q['rank']==rank
            assert q['library_sha256']==NATIVE and q['latest_protocol_binding_r290']['source_manifest_SHA256']==g['owner_manifest_SHA256']
            assert q['C2_constructor_rows']==q['constructor_logical_census']['rows']==2
            assert q['actual_layer_indices']==list(range(bounds[stage],bounds[stage+1]))
            assert q['completed_pairs']==6 and q['completed_logical_requests']==12
            assert len(q['pairs'])==6
            sig=[]
            for i,pair in enumerate(q['pairs']):
                assert pair['index']==i and pair['enabled']==(i in [1,3,4])and pair['warmup']==(i<2)
                assert len(pair['requests'])==2 and pair['decode_defer_fixed_both_arms']is True
                hit=pair['prefill_consumption'];assert hit['candidate']==CANDIDATE and hit['qualified'] and hit['enabled']==pair['enabled']
                assert (hit['complete_long_W4_groups']==5 and hit['short_advances']>0)if pair['enabled']else hit['short_advances']==hit['complete_long_W4_groups']==0
                for row,request in enumerate(pair['requests']):
                    assert request['row']==row and request['normal_reset_all32']and request['natural_EOS']and request['ordinary_budget']==4096
                    assert request['prompt_tokens']==[479,8900][row]and request['tokens']==len(request['generated_ids'])<=4096
                    sig.append((request['row'],request['prompt_IDs_SHA256'],request['generated_IDs_SHA256'],request['tokens']))
            signatures.append(sig);owners.append(dict(stage=stage,rank=rank,path=str(p),SHA256=sha(p),normal=True))
            if(stage,rank)==(3,0):tail=q
    assert all(s==signatures[0]for s in signatures),'All32 per-pair token/row/past/EOS signatures differ'
    client=Path(root)/'results'/('GLM-r361-client-'+g['run'])/'result.json';q=visible_json(client)
    assert q['candidate']==CANDIDATE and q['decode_defer_fixed']is True
    assert q['passed']and q['HTTP_requests']==q['logical_requests']==q['attempted_HTTP']==q['completed_HTTP']==q['quality_passed_HTTP']==12 and len(q['pairs'])==6
    assert tail['normal_HTTP_shutdown']and not tail['cleanup']['errors']and not tail['cleanup']['forced_cleanup']
    return dict(owner_reports=owners,stage_logs=logs,all32_token_EOS_signatures_equal=True,client_SHA256=sha(client),
                cost_audit_required=True,onlineSLO_qualified=False,old719_gain_inherited=False)


def execute(root,g,backend,clock=time.time,sleep=time.sleep):
    root=Path(root);watch=root/'results'/('GLM-r361-watch-'+g['run']+'.json');cancel=root/'control'/('GLM-r361-cancel-'+g['run']+'.json')
    assert not watch.exists()and not cancel.exists(),'Existing run must be followed, never replayed'
    q=dict(completed=False,passed=False,run=g['run'],submissions=[],actual_stages={},publication_commit=g['publication_commit'],GPU_cards=32,
           owner_manifest_SHA256=g['owner_manifest_SHA256'],controller_manifest_SHA256=g['controller_manifest_SHA256'],state='PREPARING_SINGLE_RESIDENCY',no_retry_reload=True)
    def save():
        q['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=watch.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(watch)
    with watch.open('x')as f:json.dump(q,f)
    epoch=None;unresolved={}
    try:
        backend.parents();assert all(backend.account(s)is None for s in range(4))
        backend.no_foreign();q['fresh_before']=backend.probe('before');backend.no_foreign();save()
        epoch=clock();q['original_GPU_start_unix']=epoch
        for stage in range(4):q['submissions'].append(backend.start(stage,epoch,g['owner_manifest_SHA256']));save()
        while clock()<epoch+GPU_SECONDS:
            if cancel.exists():raise InterruptedError('Original whole-set cancellation requested')
            observations={s:backend.observe(s)for s in range(4)};q['observations']=observations
            q['actual_stages']={s:o['row'][0]for s,o in observations.items()if o['row']};save()
            for s,o in observations.items():
                if(o['srun_exit']not in [None,0])or(terminal(o)and o['row'][2:4]!=['COMPLETED','0:0']):
                    raise RuntimeError('Stage%d failed; stop all four existing handles'%s)
                if(o['srun_exit']is not None and not terminal(o))or(terminal(o)and o['srun_exit']is None):
                    unresolved.setdefault(s,clock())
                    if clock()-unresolved[s]>90:raise TimeoutError('Exact stage accounting/srun visibility unresolved; no replay')
                else:unresolved.pop(s,None)
            if all(success(o)for o in observations.values()):break
            sleep(3)
        else:raise TimeoutError('Original7200s GPU bound exhausted')
        q['terminal']={s:o['row']for s,o in observations.items()};q['post']=backend.probe('post');q['post_empty']=True
        q['evidence']=collect(root,backend,g);q.update(completed=True,passed=True,state='PREFILL_INTERLEAVE_MODEL_PASS_PENDING_FULL_TIMELINE_AUDIT')
    except BaseException as error:
        q.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),state='FAILED_NO_RETRY_RELOAD')
        if not cancel.exists():
            with cancel.open('x')as f:json.dump(dict(run=g['run'],cancelled=True,error_type=type(error).__name__),f)
        if backend.processes:
            q['forced_cleanup']=backend.cancel();until=clock()+90
            while clock()<until:
                obs={s:backend.observe(s)for s in backend.processes};q['terminal_observations']=obs
                if all(terminal(o)and o['srun_exit']is not None for o in obs.values()):break
                sleep(1)
            try:q['post']=backend.probe('failure-post');q['post_empty']=True
            except BaseException as post:q['post_error']=repr(post)
        q['completed']=True
    finally:save()
    return q


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(Path(a.execution_go).read_bytes());verify(R,D,g)
    assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
    q=execute(R,g,Backend(R,D,g['run']));print(json.dumps({k:q.get(k)for k in ['completed','passed','state','actual_stages']}));return 0 if q['passed']else 1


if __name__=='__main__':raise SystemExit(main())
