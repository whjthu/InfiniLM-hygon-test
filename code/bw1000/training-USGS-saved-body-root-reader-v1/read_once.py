"""Exact saved small artifacts only; no HTTP, imports of old getter, or raw stdout."""
import argparse
import json
from pathlib import Path
from saved_parser import audit,canonical,js,need,sha

S=Path(__file__).resolve().parent;G=S.parents[2]
OUTPUT=G/'local-only/training-USGS-saved-body-root-reader-v1-original'


def consume():
    b=json.loads((S/'BINDING.json').read_bytes());run=G/b['run'];raws={}
    for name,pin in b['files'].items():
        p=run/name;need(p.is_file() and not p.is_symlink(),'Saved artifact missing/symlink')
        before=p.stat();raw=p.read_bytes();after=p.stat()
        need(before.st_size==after.st_size==pin['bytes'] and before.st_mtime_ns==after.st_mtime_ns
            and sha(raw)==pin['SHA256'],'Saved artifact differs/unstable')
        raws[name]=raw
    p=G/b['contract']['path'];raw=p.read_bytes()
    need(len(raw)==b['contract']['bytes'] and sha(raw)==b['contract']['SHA256'],'Original contract changed')
    c=json.loads(raw);intent=json.loads(raws['intent.json']);outcome=json.loads(raws['outcome.json']);worker=json.loads(raws['worker-result-private.json'])
    for obj in [intent,outcome]:
        need(obj['request']==b['request'] and obj['publication']==b['publication']
            and obj['source_manifest_SHA256']==b['source_manifest_SHA256'],'Original request identity differs')
    job=outcome['job'];need(outcome['known_terminal'] is True and job['returncode']==1 and job['timed_out'] is False,'Original producer failure not known terminal')
    need(job['WindowsJobObject_limits_verified'] is True and job['worker_CPU_count']==2 and job['job_CPU_affinity_verified'] is True
        and job['job_commit_limit_bytes']==job['process_commit_limit_bytes']==2147483648
        and job['deadline']==intent['deadline']==outcome['deadline'] and intent['deadline']-intent['epoch']==120,'Actual original Job/epoch differs')
    need(worker['passed'] is False and worker['source_accessible'] is False and worker['error_type']=='KeyError' and worker['error']=="'count'"
        and worker['actual_GET_attempts']==1 and worker['actual_body_bytes']==len(raws['original-GeoJSON-private.json'])
        and worker['charts_generated']==worker['QA_generated']==worker['model_calls']==0,'Original failure must remain unchanged')
    progress=[json.loads(line) for line in raws['progress.jsonl'].splitlines()]
    need([p['phase'] for p in progress]==['GET.enter','GET.headers_observed','GET.headers_qualified','GET.body','failure'],'Saved actual progress differs')
    need(progress[0]['attempt']==1 and progress[1]['status']==progress[2]['status']==200
        and progress[1]['Content_Type'].split(';')[0]=='application/json' and progress[2]['declared_length'] is None
        and progress[3]['bytes']==len(raws['original-GeoJSON-private.json']) and progress[4]['error_type']=='KeyError','Transport receipt differs')
    parsed=js(raws['original-GeoJSON-private.json']);summary=audit(parsed,c)
    need('count' not in parsed['metadata'] and summary['metadata_count'] is None and summary['metadata_count_present'] is False,'No count injection or limit substitution')
    need(summary['actual_event_slots']==len(parsed['features'])==50,'Actual feature denominator differs')
    OUTPUT.mkdir()
    private=canonical(summary);(OUTPUT/'source-records-private.json').write_bytes(private)
    result=dict(passed=summary['all_source_numeric_slots_qualified'],state='SAVED_BODY_NUMERIC_SOURCE_ONLY',original_request=b['request'],
        producer_passed=False,producer_source_accessible=False,producer_returncode=1,producer_error_type='KeyError',producer_error="'count'",new_GETs=0,
        transport_HTTP_status=200,actual_original_GETs=1,body_bytes=len(raws['original-GeoJSON-private.json']),body_SHA256=sha(raws['original-GeoJSON-private.json']),
        metadata_count=None,metadata_count_present=False,actual_event_slots=summary['actual_event_slots'],maximum_requested_slots=50,
        qualified_numeric_slots=summary['qualified_numeric_slots'],unqualified_retained_slots=summary['unqualified_retained_slots'],
        event_alias_groups=summary['event_alias_groups'],geographic_groups=summary['geographic_groups'],sample_limit_reached=summary['sample_limit_reached'],
        whole_catalog_window_complete=summary['whole_catalog_window_complete'],api_version=summary['api_version'],generated_epoch_ms=summary['generated_epoch_ms'],
        private_summary_SHA256=sha(private),binding_SHA256=sha((S/'BINDING.json').read_bytes()),reader_manifest_SHA256=sha((S/'MANIFEST.json').read_bytes()),
        original_artifacts=b['files'],charts_generated=0,QA_generated=0,Dataset_GO=False,training_GO=False,GPU_GO=False)
    (OUTPUT/'result.json').write_bytes(canonical(result));print(json.dumps(result,sort_keys=True));return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root-go',action='store_true');args=p.parse_args()
    need(args.root_go,'Source prepared only; root must review once');raise SystemExit(consume())
