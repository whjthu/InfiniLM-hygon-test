"""One <=2MiB official metadata request inside actual2CPU/2GiB/single120s Job."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from job_guard import current_affinity
from metadata import canonical,get_once,inventory,need,strict_json,url_gate


def main(directory,epoch,deadline):
    start=time.monotonic();end=start+max(0,deadline-time.time())
    def remaining():
        left=min(deadline-time.time(),end-time.monotonic())-8;need(left>0,'Original120s epoch exhausted');return left
    while not (directory/'job-gate.json').exists():remaining();time.sleep(.02)
    gate=json.loads((directory/'job-gate.json').read_bytes())
    need(deadline-epoch==120 and gate['deadline']==deadline and gate['memory_limit_bytes']==2147483648 and gate['CPU_count']==2
        and gate['job_affinity_verified'] is True and gate['worker_affinity_mask']==current_affinity()['process_mask'],'Actual Job/epoch/affinity differs')
    source=Path(__file__).parent;contract=json.loads((source/'CONTRACT.json').read_bytes());ledger=json.loads((source/'WEB_ATTEMPTS.json').read_bytes())
    result=dict(passed=False,epoch=epoch,deadline=deadline,planned_GETs=1,actual_GET_attempts=0,actual_metadata_bytes=0,
        Dataset_GO=False,training_GO=False,GPU_GO=False,actual_images=0,actual_parquet_bytes=0,actual_blind_gold_read=0)
    def progress(phase,details):
        if phase=='GET.enter':result['actual_GET_attempts']+=1
        if phase=='GET.body':result['actual_metadata_bytes']=details['actual_bytes']
        with (directory/'progress.jsonl').open('ab') as stream:stream.write(canonical(dict(phase=phase,time=time.time(),**details)))
    try:
        url_gate(contract,ledger);remaining()
        raw,http=get_once(contract,ledger,directory/'repository-metadata-private.json',remaining,progress)
        summary=inventory(strict_json(raw),contract);(directory/'file-inventory-private.json').write_bytes(canonical(summary))
        result.update(passed=True,state=summary['state'],http=http,parquet_file_count=6,
            file_inventory_SHA256=hashlib.sha256((directory/'file-inventory-private.json').read_bytes()).hexdigest(),Range_qualified=False)
    except Exception as error:
        result.update(state='KNOWN_FAILURE_METADATA_ONLY_NO_RETRY_OR_BODY_FALLBACK',error_type=type(error).__name__,error=str(error))
        progress('failure',dict(error_type=type(error).__name__,error=str(error)))
    result['wall_seconds']=time.monotonic()-start;(directory/'worker-result.json').write_bytes(canonical(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();raise SystemExit(main(a.run,a.epoch,a.deadline))
