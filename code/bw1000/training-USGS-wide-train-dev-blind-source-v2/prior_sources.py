"""Load exactly the frozen250 old numeric records and2724 closure edges; no questions/gold."""
from datetime import datetime
import json
from chart_contracts import need,sha

def stable(path,pin):
    need(path.is_file() and not path.is_symlink(),'EXACT_PRIOR_REGULAR_FILE')
    a=path.stat();need(a.st_size==pin['bytes']<=256*1024,'BOUNDED_PRIOR_METADATA')
    raw=path.read_bytes();b=path.stat()
    need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
        and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'EXACT_PRIOR_METADATA_PIN')
    return json.loads(raw)

def load(g,plan):
    pins=plan['prior_sources'];old=stable(g/pins['old50']['path'],pins['old50']);new=stable(g/pins['new200']['path'],pins['new200']);closure=stable(g/pins['closure']['path'],pins['closure'])
    need(old['actual_event_slots']==50 and new['actual_event_slots']==200 and old['all_source_numeric_slots_qualified'] is True and new['all_source_numeric_slots_qualified'] is True,'OLD250_QUALIFIED_SOURCE_DENOMINATOR')
    rows=old['rows']+new['rows'];need(len(rows)==250 and all(r['status']=='QUALIFIED_SOURCE_NUMERIC_RECORD' and not r['errors'] and not r['null_fields'] for r in rows),'OLD250_NO_REGENERATION_OR_REPLACEMENT')
    start=int(datetime.fromisoformat(plan['fixed_start_UTC']).timestamp()*1000);end=int(datetime.fromisoformat(plan['fixed_end_UTC']).timestamp()*1000)
    need(all(type(r['time_UTC_epoch_ms'])is int and not start<=r['time_UTC_epoch_ms']<=end for r in rows),'NEW_FIXED_WINDOW_OVERLAPS_KNOWN_250')
    edges=closure['closure_edges'];need(closure['old_slots']==50 and closure['new_slots']==200 and len(edges)==2724 and closure['passed'] is True,'ORIGINAL2724_CLOSURE_IDENTITY')
    need(len({(e['a'],e['b']) for e in edges})==2724 and all(type(e['a'])is int and type(e['b'])is int and 0<=e['b']<e['a']<250 and e['reasons'] and set(e['reasons'])<={'alias','coarse_region','near72h500km'} for e in edges),'ORIGINAL_EDGE_SCHEMA')
    return rows,edges
