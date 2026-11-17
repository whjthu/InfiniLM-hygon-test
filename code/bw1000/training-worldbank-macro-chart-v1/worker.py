"""Private data acquisition/build worker, entered only after real WindowsJob gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from build import build,renderer_preflight
from contracts import canonical,need,plan_check,parse_indicator,parse_countries,strict_json
from fetch import fetch_one
from job_guard import current_affinity


def run(directory,epoch,deadline):
    start=time.monotonic();end=start+max(0,deadline-time.time())
    def check():
        left=min(deadline-time.time(),end-time.monotonic())-8
        need(left>0,'Original300s epoch exhausted');return left
    while not (directory/'job-gate.json').exists():check();time.sleep(.02)
    gate=json.loads((directory/'job-gate.json').read_bytes())
    need(gate['deadline']==deadline and gate['memory_limit_bytes']==2147483648 and gate['CPU_count']==2
        and gate['job_affinity_verified'] and current_affinity()['process_mask']==gate['worker_affinity_mask'],'Actual Job/affinity gate differs')
    plan=json.loads((Path(__file__).parent/'PLAN.json').read_bytes());plan_check(plan)
    need(deadline-epoch==300 and time.time()>=epoch-1,'Wrong single epoch')
    result=dict(epoch=epoch,deadline=deadline,planned_facts=504,planned_charts=72,planned_QA=648,attempted_GETs=0,
        completed_GETs=0,actual_inputs=0,actual_charts=0,passed=False,Dataset_GO=False,training_GO=False,GPU_GO=False)
    def progress(phase,details):
        with (directory/'progress.jsonl').open('ab') as stream:stream.write(canonical(dict(phase=phase,time=time.time(),**details)))
    try:
        progress('renderer.preflight',{})
        # Deliberate before-GET hard prerequisite. No installation/fallback here.
        renderer=renderer_preflight(plan);check();progress('renderer.ready',renderer)
        raw_directory=directory/'raw';raw_directory.mkdir();documents={};http={}
        # Metadata first: do not spend fact responses if required source fields are absent later.
        names=['indicator_0','indicator_1','indicator_2','countries','facts_train','facts_dev','facts_blind']
        for name in names:
            check();result['attempted_GETs']+=1
            progress('GET.enter',dict(endpoint=name,attempt=result['attempted_GETs']))
            documents[name],http[name]=fetch_one(plan['endpoints'][name],raw_directory/(name+'.json'),check,
                lambda phase,details:progress('GET.'+phase,dict(endpoint=name,**details)))
            result['completed_GETs']+=1
            if 'indicator' in plan['endpoints'][name]:parse_indicator(strict_json(documents[name]),plan['endpoints'][name]['indicator'],plan)
            if name=='countries':parse_countries(strict_json(documents[name]),plan)
        need(sum(len(x) for x in documents.values())<=8388608,'Raw total cap exceeded')
        manifest=build(documents,plan,directory/'derived',check,renderer)
        result.update(passed=True,state='DATA_BUILT_NATIVE_READABILITY_AND_MODEL_PENDING',actual_inputs=manifest['actual_inputs'],actual_charts=manifest['actual_charts'],
            actual_nonmissing_facts=manifest['actual_nonmissing_facts'],missing_facts=manifest['missing_facts'],http=http,
            manifest_SHA256=hashlib.sha256((directory/'derived/manifest-private.json').read_bytes()).hexdigest())
    except Exception as error:
        result.update(state='FAILED_KEEP_ALL504_FACT_72_CHART_648_QA_SLOTS',error_type=type(error).__name__,error=str(error))
        progress('failure',dict(error_type=type(error).__name__,error=str(error)))
    result['wall_seconds']=time.monotonic()-start
    (directory/'worker-result.json').write_bytes(canonical(result));return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();raise SystemExit(run(a.run,a.epoch,a.deadline))
