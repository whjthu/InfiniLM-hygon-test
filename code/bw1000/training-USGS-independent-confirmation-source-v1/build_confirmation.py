"""One fixed confirmation split; derive old exact question kinds, no inference/scoring."""
from pathlib import Path
from chart_contracts import canonical,sha,questions,displayed_rows,need
from render import draw


def build(selection,plan,out,renderer,remaining,progress,source):
    need(selection['passed'] is True and not out.exists(),'FIVE_NEW_WINDOWS_REQUIRED')
    out.mkdir(mode=0o700)
    # Group membership/window selection is durable before QA/gold construction.
    declaration={k:v for k,v in selection.items() if k!='charts'}
    declaration['charts']=[dict(id=c['id'],group=c['group'],split='confirmation',event_ids=[r['id'] for r in c['rows']]) for c in selection['charts']]
    (out/'declaration-private.json').write_bytes(canonical(declaration))
    inputs=[];labels=[];media={};chart_records=[];total=0;seen=set()
    for i,c in enumerate(selection['charts']):
        remaining();qs=questions(c);record=dict(id=c['id'],group=c['group'],planned_QA=5,status='PENDING')
        try:
            body=draw(c,plan,renderer);h=sha(body);need(h not in seen,'NO_DUPLICATE_MEDIA_OR_BACKFILL');seen.add(h)
            name='confirmation/media/'+str(i).zfill(3)+'.png';path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
            pin=dict(bytes=len(body),SHA256=h,width=512,height=512,name=name);media[name]=pin;total+=len(body)
            need(total<=plan['PNG_total_cap_bytes'],'FIVE_PNG_BUDGET')
            record.update(status='QUALIFIED',image=pin,display_rows_SHA256=sha(canonical(displayed_rows(c))),source_events=len(c['rows']))
            for q in qs:
                inputs.append(dict(id=q['id'],chart_group=c['id'],source_group=c['group'],messages=[dict(role='user',content='<image>\n'+q['question'])],images=[name]))
                labels.append(dict(q,data_status='QUALIFIED'))
        except Exception as e:
            record.update(status='FAILED',error_type=type(e).__name__)
            for q in qs:labels.append(dict(q,data_status='FAILED',answer=None))
        chart_records.append(record);progress('chart_terminal',dict(completed_chart_slots=len(chart_records),planned_charts=5,completed_QA_slots=len(labels),planned_QA=25))
    pins={}
    for name,rows in [('confirmation/inputs.jsonl',inputs),('confirmation/labels-private.jsonl',labels)]:
        raw=b''.join(canonical(r) for r in rows);p=out/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw);pins[name]=dict(bytes=len(raw),SHA256=sha(raw),rows=len(rows))
    credits=dict(publisher='U.S. Geological Survey Earthquake Catalog',URL=plan['URL'],source_body_SHA256=source['body_SHA256'],generated_epoch_ms=source['generated_epoch_ms'],api_version=source['api_version'],
        rights='USGS-produced numerical data U.S.publicdomain/free use with attribution; source networks retained. No worldwideCC0 or thirdparty image rights claim.',rights_URL='https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted',
        modifications='New512x512 local plots, displayedDecimal2 and frozen5QA kinds; not official USGS images, not energy/risk comparisons.',endorsement=False)
    (out/'CREDITS.json').write_bytes(canonical(credits))
    manifest=dict(state='INDEPENDENT_CONFIRMATION_DATA_ONLY',planned_charts=5,planned_QA=25,actual_charts=sum(c['status']=='QUALIFIED' for c in chart_records),actual_inputs=len(inputs),retained_QA=len(labels),
        new_source_slots=selection['new_slots'],excluded_new_slots=selection['excluded_count'],eligible_groups=selection['eligible_groups'],selected_groups=selection['selected_group_count'],selected_events=selection['selected_event_count'],
        declaration_SHA256=sha((out/'declaration-private.json').read_bytes()),outputs=pins,media=media,charts=chart_records,PNG_bytes=total,source=source,
        old_source_pin=plan['old_source_pin'],same_frozen_closure_overlap=0,no_absolute_leakage_guarantee=True,source_window_complete='UNKNOWN_LIMITED_QUERY',renderer=renderer,
        visual_QA='PENDING',native_processor='NOT_RUN',Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False)
    need(len(labels)==25,'FULL25_QA_DENOMINATOR');(out/'manifest-private.json').write_bytes(canonical(manifest));return manifest
