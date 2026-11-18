"""One new bounded public-source/data qualification. No model, SSH, GPU or training."""
import argparse,json,re,subprocess,sys,time,uuid,math
from pathlib import Path
from chart_contracts import canonical,need,sha
from clock_ipc import validate_gate
from job_guard import current_affinity,guarded_run
S=Path(__file__).resolve().parent;G=S.parents[2];RUN=G/'local-only/training-USGS-independent-confirmation-source-v1-original'


def stable(path,pin):
    need(path.is_file() and not path.is_symlink(),'EXACT_OLD_SOURCE_REGULAR_FILE');a=path.stat();raw=path.read_bytes();b=path.stat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns) and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'EXACT_OLD_SOURCE_PIN');return raw


def worker(epoch,deadline):
    started=time.monotonic();mono=started+max(0,deadline-time.time())
    def remaining():
        left=min(deadline-time.time(),mono-time.monotonic())-8;need(left>0,'ORIGINAL120_EPOCH_EXHAUSTED');return left
    while not (RUN/'job-gate.json').exists():remaining();time.sleep(.02)
    validate_gate((RUN/'job-gate.json').read_bytes(),epoch,deadline,current_affinity()['process_mask'])
    (RUN/'worker-enter.lock').open('x').close();plan=json.loads((S/'PLAN.json').read_bytes())
    result=dict(passed=False,epoch=epoch,deadline=deadline,planned_charts=5,planned_QA=25,planned_future_predictions=75,actual_GETs=0,new_body_bytes=0,
        actual_charts=0,actual_QA=0,Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False,model_calls=0,native_processor_calls=0)
    def progress(phase,info):
        if phase=='GET.enter':result['actual_GETs']+=1
        if phase=='GET.body':result['new_body_bytes']=info['bytes']
        with (RUN/'progress.jsonl').open('ab') as f:f.write(canonical(dict(phase=phase,time=time.time(),**info)))
    try:
        need(plan['root_GO'] is plan['CPU_GO'] is plan['source_GET_GO'] is True,'PUBLISHED_DATA_CPU_ADOPTION_REQUIRED')
        old=json.loads(stable(G/plan['old_source_path'],plan['old_source_pin']))['rows'];need(len(old)==50,'ORIGINAL_OLD50_DENOMINATOR')
        from render import renderer_preflight
        renderer=renderer_preflight(plan) # Actual qualified renderer/font before any GET.
        from source_http import get_once
        from saved_parser import js,audit
        from select_groups import select
        from build_confirmation import build
        remaining();raw=get_once(RUN/'source-GeoJSON-private.json',plan,remaining,progress)
        parsed=audit(js(raw),plan);(RUN/'source-numerics-private.json').write_bytes(canonical(parsed))
        result.update(new_source_slots=parsed['actual_event_slots'],source_body_SHA256=sha(raw),all_source_numeric_slots_qualified=parsed['all_source_numeric_slots_qualified'])
        need(parsed['all_source_numeric_slots_qualified'] is True,'SOURCE_SLOT_FAILURE_RETAINED_NO_SUBSTITUTION')
        selected=select(old,parsed['rows'],plan)
        (RUN/'closure-selection-private.json').write_bytes(canonical(selected));progress('closure_frozen',dict(new_slots=selected['new_slots'],excluded_count=selected['excluded_count'],eligible_groups=selected['eligible_groups'],selected_groups=selected['selected_group_count'],planned_charts=5,planned_QA=25))
        result.update(eligible_groups=selected['eligible_groups'],excluded_new_source_slots=selected['excluded_count'],selected_groups=selected['selected_group_count'])
        need(selected['passed'] is True,'INSUFFICIENT_INDEPENDENT_GROUPS_STOP_NO_PARAMETER_CHANGE')
        source=dict(body_SHA256=sha(raw),api_version=parsed['api_version'],generated_epoch_ms=parsed['generated_epoch_ms'],query_URL_SHA256=sha(plan['URL'].encode()))
        manifest=build(selected,plan,RUN/'dataset',renderer,remaining,progress,source);remaining()
        total=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file());need(total<=plan['total_new_local_bytes_cap'],'NEW_LOCAL_ARTIFACTS20M_CAP')
        result.update(passed=manifest['actual_charts']==5 and manifest['actual_inputs']==manifest['retained_QA']==25,state='CONFIRMATION_DATA_ONLY_PENDING_INDEPENDENT_NATIVE_QUALIFICATION',
            actual_charts=manifest['actual_charts'],actual_QA=manifest['retained_QA'],actual_user_inputs=manifest['actual_inputs'],new_local_bytes=total,manifest_SHA256=sha((RUN/'dataset/manifest-private.json').read_bytes()))
    except Exception as error:
        result.update(state='KNOWN_FAILED_NO_NEW_WINDOW_RETRY_OR_BACKFILL',error_type=type(error).__name__,error=str(error)[:512]);progress('failure',dict(error_type=type(error).__name__))
    result['wall_seconds']=time.monotonic()-started
    (RUN/'worker-result.json').write_bytes(canonical(result));return 0 if result['passed'] else 1


def validate_origin(epoch,now):
    need(type(epoch) in (int,float) and type(now) in (int,float)
         and math.isfinite(epoch) and math.isfinite(now) and 0<epoch<=now<epoch+120,
         'FINITE_ORIGINAL120_ORIGIN_NO_FUTURE_OR_EXPIRED')
    return epoch+120


def controller(pub,claim,go,origin=None):
    epoch=time.time() if origin is None else origin;deadline=validate_origin(epoch,time.time());mono=time.monotonic()+max(0,deadline-time.time())
    def remaining():
        left=min(deadline-time.time(),mono-time.monotonic())-8;need(left>0,'ORIGINAL120_EPOCH_EXHAUSTED');return left
    need(go and re.fullmatch('[0-9a-f]{40}',pub or '') and re.fullmatch('[0-9a-f]{64}',claim or ''),'ROOT_PUBLISHED_SOURCE_ADMISSION')
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),*a],cwd=G,timeout=min(10,remaining()))
    need(git('remote','get-url','origin').decode().strip()=='git@github.com:whjthu/InfiniLM-hygon-test.git','EXPECTED_ORIGIN')
    need(git('rev-parse','HEAD').decode().strip()==pub,'PUBLISHED_HEAD')
    need(git('ls-remote','origin','refs/heads/bw1000-experiments').decode().split()==[pub,'refs/heads/bw1000-experiments'],'ORDINARY_REMOTE_HEAD')
    need(not git('status','--porcelain','--untracked-files=all','--',S.relative_to(G).as_posix()).strip() and not git('diff','--cached','--name-only').strip(),'OWN_DIRTY_OR_INDEX')
    need(sha((S/'CLAIM.json').read_bytes())==claim,'CLAIM_PIN')
    for name,pin in json.loads((S/'MANIFEST.json').read_bytes())['files'].items():
        body=(S/name).read_bytes();need(len(body)==pin['bytes'] and sha(body)==pin['SHA256'] and git('show',pub+':'+(S/name).relative_to(G).as_posix())==body,'PUBLISHED_EXACT_SOURCE')
    need(git('show',pub+':'+(S/'MANIFEST.json').relative_to(G).as_posix())==(S/'MANIFEST.json').read_bytes(),'PUBLISHED_MF')
    plan=json.loads((S/'PLAN.json').read_bytes());adopt=json.loads((S/'CLAIM.json').read_bytes())
    need(adopt['root_GO'] is adopt['CPU_GO'] is adopt['source_GET_GO'] is True and adopt['GPU_GO'] is adopt['training_GO'] is False,'ROOT_DATA_ONLY_ADOPTION')
    remaining();need(not RUN.exists(),'EXCLUSIVE_ORIGINAL_NAMESPACE_NO_REPLAY');RUN.mkdir()
    intent=dict(request=uuid.uuid4().hex,publication=pub,claim_SHA256=claim,source_manifest_SHA256=sha((S/'MANIFEST.json').read_bytes()),epoch=epoch,deadline=deadline,planned_GETs=1,maximum_source_slots=200,body_cap_bytes=1048576,planned_charts=5,planned_QA=25,root_GO=True)
    (RUN/'intent.json').write_bytes(canonical(intent))
    job=guarded_run([sys.executable,'-B','-X','utf8',str(S/'once.py'),'--worker','--epoch',str(epoch),'--deadline',str(deadline)],RUN/'job-gate.json',RUN,deadline,mono)
    (RUN/'outcome.json').write_bytes(canonical(dict(**intent,known_terminal=True,job=job,Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False)))
    print(json.dumps(dict(request=intent['request'],known_terminal=True,returncode=job['returncode'],timed_out=job['timed_out'])));return job['returncode']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--epoch',type=float);p.add_argument('--deadline',type=float);p.add_argument('--origin',type=float)
    p.add_argument('--publication');p.add_argument('--claim-sha256');p.add_argument('--root-go',action='store_true');a=p.parse_args()
    raise SystemExit(worker(a.epoch,a.deadline) if a.worker else controller(a.publication,a.claim_sha256,a.root_go,a.origin))
