"""The new data identity and complete failure denominator; standard library only."""
import hashlib
import json

ROWS=72


def need(ok,code):
    if not ok:raise ValueError(code)


def row_sha(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n').hexdigest()


def require_row(row,expected,ordinal):
    need(expected['ordinal']==ordinal and row_sha(row)==expected['row_SHA256'],'NEW72_ORDERED_INPUT_IDENTITY')
    need(type(row.get('id')) is str and hashlib.sha256(row['id'].encode()).hexdigest()==expected['source_ID_SHA256'],'NEW72_SOURCE_ID')
    need(len(row['images'])==1 and row['images'][0]==expected['image'],'NEW72_COMPLETE_IMAGE_IDENTITY')
    need([x['role'] for x in row['messages']]==['user','assistant'] and row['messages'][0]['content']
         and row['messages'][1]['content'].startswith('FINAL: '),'NEW72_NATIVE_TRAIN_MESSAGES')


def validate_progress(p):
    need(p['planned_rows']==ROWS and type(p['failed']) is bool,'NEW72_DENOMINATOR')
    c,a=p['completed_rows'],p['attempted_rows']
    need(type(c) is type(a) is int and 0<=c<=a<=ROWS,'NEW72_COUNTS')
    need(p['missing_rows']==ROWS-c and p['unattempted_rows']==ROWS-a,'NEW72_MISSING_COUNTS')
    need(p['failed_rows']==(1 if p['failed'] and a>c else 0),'NEW72_FAILED_ROW_COUNT')
    ea,ec,ca,cc=(p[k] for k in ('encode_calls_attempted','encode_calls_completed','collator_calls_attempted','collator_calls_completed'))
    need(all(type(v) is int for v in (ea,ec,ca,cc)) and 2*c<=ec<=ea<=2*a and c<=cc<=ca<=a,'NEW72_NATIVE_CALL_COUNTS')
    if not p['failed'] and c==ROWS:
        need(a==ROWS and ea==ec==144 and ca==cc==72,'NEW72_COMPLETE_NATIVE_CALLS')
    return p
