"""Only new query metadata count variants; synthetic, no saved actual consumption."""
from copy import deepcopy
from saved_parser import audit,js
import json


def run():
    c={'fixed_start_UTC':'2026-10-01T00:00:00+00:00','fixed_end_UTC':'2026-10-08T00:00:00+00:00'}
    base=js(b'{"type":"FeatureCollection","metadata":{"status":200,"api":"2.8.4","generated":1794943695000,"limit":50,"offset":1},"features":[{"type":"Feature","id":"xx1","properties":{"type":"earthquake","mag":4.50,"magType":"mb","time":1790812800000,"updated":1790812800001,"ids":",xx1,","net":"xx","sources":",xx,"},"geometry":{"type":"Point","coordinates":[120.25,-12.5,-1.20]}}]}')
    absent=audit(base,c);assert absent['actual_event_slots']==1 and absent['metadata_count'] is None and not absent['metadata_count_present'] and absent['rows'][0]['values']['depth_km']=='-1.20'
    present=deepcopy(base);present['metadata']['count']=1;assert audit(present,c)['metadata_count']==1
    null=deepcopy(base);null['features'][0]['properties']['mag']=None
    assert audit(null,c)['actual_event_slots']==1 and audit(null,c)['unqualified_retained_slots']==1
    rejected=[]
    for value,name in [(50,'limit_is_not_count'),(None,'present_null'),(True,'present_bool'),('1','present_string'),(0,'mismatched_count')]:
        x=deepcopy(base);x['metadata']['count']=value
        try:audit(x,c)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    return dict(passed=True,normal=3,rejected=len(rejected),rejected_cases=rejected,HTTP_GETs=0,actual_saved_body_consumptions=0,old_scientific_tests_rerun=0)

if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
