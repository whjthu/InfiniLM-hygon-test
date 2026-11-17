"""Independent saved-artifact audit; no build/render/network/model imports or blind stdout."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

G=Path(__file__).resolve().parents[3]
SOURCE=G/'code/bw1000/training-USGS-reported-event-charts-v1'
RUN=G/'local-only/training-USGS-reported-event-charts-v1-original'
DATA=RUN/'dataset'
OUT=G/'local-only/training-USGS-reported-event-charts-v1-independent-audit/artifact-only-final'
REQUEST='e0746bb9b33f424db697e4c1c2325b54'
PUBLICATION='aced02165cc139130591f41af433e906912b53b6'
PINS={'intent.json':(442,'5c756bd39c774b9bbb0b2751466d105e7cae8d98c7853f32460fd595d44c0703'),
'outcome.json':(1146,'c8fe7c606ae5d3166dfc46dc6f9d75103fb17567b4dbddac39069018ec352131'),
'worker-result.json':(765,'4793f1642eedb88cab69d64f63c4cac002fdc0d480dd2806a79b08898432e25b')}


def need(ok,msg):
    if not ok:raise ValueError(msg)

def canonical(x):return (json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p,pin=None):
    need(p.is_file() and not p.is_symlink(),'Missing/symlink exact artifact')
    a=p.stat();b=p.read_bytes();c=p.stat();need(a.st_size==c.st_size==len(b) and a.st_mtime_ns==c.st_mtime_ns,'Artifact unstable')
    if pin:need(len(b)==pin[0] and sha(b)==pin[1],'Artifact identity differs')
    return b

def rounded(x):
    d=Decimal(x);need(d.is_finite(),'Nonfinite source display')
    d=d.quantize(Decimal('.01'),rounding=ROUND_HALF_UP);return format(Decimal('0.00') if d==0 else d,'.2f')

def near(a,b):
    if abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms'])>259200000:return False
    lat1,lat2=[float(r['values']['latitude'])*math.pi/180 for r in [a,b]]
    delta=(float(a['values']['longitude'])-float(b['values']['longitude']))*math.pi/180
    hav=(1-math.cos(lat1-lat2))/2+math.cos(lat1)*math.cos(lat2)*(1-math.cos(delta))/2
    km=2*6371.0088*math.atan2(math.sqrt(max(0,min(1,hav))),math.sqrt(max(0,1-min(1,hav))))
    return km<=500

def png_identity(raw):
    need(raw[:8]==b'\x89PNG\r\n\x1a\n','PNG signature')
    pos=8;chunks=[];software=None;dims=None
    while pos<len(raw):
        need(pos+12<=len(raw),'PNG truncated header');n=struct.unpack('>I',raw[pos:pos+4])[0];kind=raw[pos+4:pos+8];end=pos+12+n
        need(end<=len(raw),'PNG truncated payload');value=raw[pos+8:pos+8+n]
        need(zlib.crc32(kind+value)&0xffffffff==struct.unpack('>I',raw[end-4:end])[0],'PNG chunk CRC')
        if kind==b'IHDR':need(not chunks and n==13,'PNG IHDR');dims=struct.unpack('>II',value[:8])
        if kind==b'tEXt' and value.startswith(b'Software\0'):software=value.split(b'\0',1)[1].decode('latin1')
        chunks.append(kind);pos=end
        if kind==b'IEND':need(n==0 and pos==len(raw),'PNG trailing data');break
    need(chunks[-1]==b'IEND' and dims==(512,512),'PNG dimensions/end')
    need(software=='training-USGS-reported-event-charts-v1','PNG Software provenance')
    return dict(width=512,height=512,Software=software,passive_png_chunk_CRC_verified=True,pixel_decode=False)

def audit():
    OUT.mkdir();intent=json.loads(read(RUN/'intent.json',PINS['intent.json']));outcome=json.loads(read(RUN/'outcome.json',PINS['outcome.json']));worker=json.loads(read(RUN/'worker-result.json',PINS['worker-result.json']))
    need(intent['request']==outcome['request']==REQUEST and intent['publication']==outcome['publication']==PUBLICATION,'Original invocation differs')
    mf=read(SOURCE/'MANIFEST.json');need(sha(mf)==intent['source_manifest_SHA256']=='5ad91c023581a3f80191d0b1ced335dd4463b9f02af84872d69557e6b1beb9c4','Source MF differs')
    for name,pin in json.loads(mf)['files'].items():read(SOURCE/name,(pin['bytes'],pin['SHA256']))
    need(sha(read(SOURCE/'CLAIM.json'))==intent['claim_SHA256']=='9e7b524cc89eef2c3a8f7ef5921e25f81c194356dbcaae34dbb558b5fc7f84c1','Actual claim differs')
    plan=json.loads(read(SOURCE/'PLAN.json'));job=outcome['job']
    need(outcome['known_terminal'] is True and job['returncode']==0 and job['timed_out'] is False and worker['passed'] is True,'Original build not successful terminal')
    need(job['deadline']==intent['deadline']==worker['deadline'] and worker['epoch']==intent['epoch'] and intent['deadline']-intent['epoch']==120,'Whole epoch differs')
    need(job['worker_CPU_count']==2 and job['WindowsJobObject_limits_verified'] is True and job['job_CPU_affinity_verified'] is True and job['worker_CPU_affinity_verified_before_gate'] is True
        and job['job_commit_limit_bytes']==job['process_commit_limit_bytes']==2147483648 and job['actual_peak_job_commit_bytes']<=2147483648,'Actual Job resources differ')
    need(intent['HTTP_GETs']==outcome['HTTP_GETs']==worker['HTTP_GETs']==worker['model_calls']==0,'Forbidden actual workload')
    mraw=read(DATA/'manifest-private.json');need(sha(mraw)==worker['manifest_SHA256']=='9423f4b9b3b40d64260f6d8a43c7c50e4595bcc887767fc47fba5db3d9145390','Private output manifest differs');m=json.loads(mraw)
    sraw=read(DATA/'split-plan-private.json');need(sha(sraw)==m['split_plan_SHA256'],'Split-plan identity differs');splitplan=json.loads(sraw)
    source=G/plan['source_path'];docs={name:json.loads(read(source/name,(pin['bytes'],pin['SHA256']))) for name,pin in plan['source_pins'].items()}
    qualification=docs['result.json'];rows=docs['source-records-private.json']['rows']
    need(qualification['passed'] is True and qualification['producer_passed'] is False and qualification['metadata_count'] is None
        and qualification['body_SHA256']==plan['original_source_body_SHA256'] and len(rows)==50,'Source qualification or preserved producer failure differs')
    need(len({r['id'] for r in rows})==50 and set(r['slot'] for r in rows)==set(range(50)),'Source denominator/slot identity')
    byid={r['id']:r for r in rows};byslot={r['slot']:r for r in rows}
    # Independent graph traversal (not builder DSU). No answer or model used in grouping.
    adjacency={r['id']:set() for r in rows};expected_edges={}
    for i,a in enumerate(rows):
        for b in rows[:i]:
            reasons=[]
            if set(a['aliases'])&set(b['aliases']):reasons.append('same_alias')
            if a['source_geo_region_key']==b['source_geo_region_key']:reasons.append('same_coarse_region')
            if near(a,b):reasons.append('near_72h_500km')
            if reasons:adjacency[a['id']].add(b['id']);adjacency[b['id']].add(a['id']);expected_edges[(a['id'],b['id'])]=reasons
    need({(e['a'],e['b']):e['reasons'] for e in splitplan['edges']}==expected_edges,'Closure edge evidence differs')
    groups=[];seen=set()
    for identifier in sorted(adjacency):
        if identifier in seen:continue
        pending=[identifier];members=[]
        while pending:
            x=pending.pop()
            if x in seen:continue
            seen.add(x);members.append(x);pending.extend(adjacency[x]-seen)
        gid=sha(canonical(sorted(members)));groups.append(dict(group=gid,event_ids=sorted(members,key=lambda x:(byid[x]['time_UTC_epoch_ms'],x)),order=sha((plan['seed']+gid).encode())))
    groups.sort(key=lambda x:(x['order'],x['group']));need(len(groups)>=3,'No isolated split groups');ntrain=max(1,int(len(groups)*.6));ndev=max(1,(len(groups)-ntrain)//2)
    expected_groups=[];expected_charts=[];event_split={}
    for i,g in enumerate(groups):
        split='train' if i<ntrain else ('dev' if i<ntrain+ndev else 'blind')
        expected_groups.append(dict(group=g['group'],event_ids=g['event_ids'],split=split))
        for identifier in g['event_ids']:event_split[identifier]=split
        for offset in range(0,len(g['event_ids']),5):expected_charts.append(dict(id='usgs:'+g['group']+':'+str(offset//5),group=g['group'],split=split,source_slots=[byid[x]['slot'] for x in g['event_ids'][offset:offset+5]]))
    need(splitplan['groups']==expected_groups and splitplan['event_split']==event_split and splitplan['charts']==expected_charts,'Fixed pre-QA split/window reconstruction differs')
    need(all(event_split[a]==event_split[b] for a,b in expected_edges),'Near/alias/region edge crosses split')
    need(len({s for c in expected_charts for s in c['source_slots']})==50 and sum(len(c['source_slots']) for c in expected_charts)==50,'Source event coverage not exactly once')
    need(len(expected_charts)==m['planned_charts']==m['actual_charts']==16 and len(expected_charts)*5==m['planned_QA']==m['retained_QA_slots']==80,'Actual chart/QA denominator differs')
    need(m['failed_charts']==0 and m['closure_groups']==len(groups)==12 and m['source_alias_groups']==50 and m['source_geographic_groups']==12,'Actual groups or chart failures differ')
    expected_questions={
        'lookup_reported_magnitude':'What is the reported magnitude number for E01? Answer with exactly two decimal places.',
        'lookup_depth':'What is the depth in km for E01? Answer with exactly two decimal places.',
        'depth_change':'Compute the last listed row depth minus E01 depth, in km. Use the displayed numbers and answer with exactly two decimal places, with a minus sign if negative.',
        'maximum_depth':'Which event label(s) have the largest displayed depth in km? Include every tie, separated by one space in table order.',
        'maximum_reported_number':'Which event label(s) have the largest displayed reported magnitude NUMBER? Compare displayed numbers only, not physical energy or risk. Include every tie, separated by one space in table order.'}
    files={};allinputs={};alllabels={};splitcounts={}
    for split in ['train','dev','blind']:
        values={}
        for name in ['inputs.jsonl','labels-private.jsonl']:
            rel=split+'/'+name;pin=m['outputs'][rel];raw=read(DATA/rel,(pin['bytes'],pin['SHA256']));items=[json.loads(line) for line in raw.splitlines()]
            need(len(items)==pin['rows'] and len({r['id'] for r in items})==len(items),'Output row identity/denominator differs');values[name]=items;files[rel]=pin
        need(len(values['inputs.jsonl'])==len(values['labels-private.jsonl']),'Input-label full denominator differs');splitcounts[split]=len(values['inputs.jsonl'])
        for r in values['inputs.jsonl']:need(r['id'] not in allinputs,'Crosssplit input duplicate');allinputs[r['id']]=(split,r)
        for r in values['labels-private.jsonl']:need(r['id'] not in alllabels,'Crosssplit label duplicate');alllabels[r['id']]=(split,r)
    kinds=Counter();split_charts=Counter();split_singletons=Counter();pnghashes=set();displayhashes=set();bytes_png=0;singletons=Counter();visual={split:[] for split in ['train','dev']}
    for record,expected in zip(m['charts'],expected_charts):
        need(all(record[k]==v for k,v in expected.items()) and record['status']=='QUALIFIED' and record['planned_QA']==5,'Chart identity/group differs')
        observed=[byslot[x] for x in expected['source_slots']];singletons[str(len(observed)==1)]+=1;split_charts[record['split']]+=1
        if len(observed)==1:split_singletons[record['split']]+=1
        shown=[dict(label='E'+str(i+1).zfill(2),date_UTC=datetime.fromtimestamp(r['time_UTC_epoch_ms']/1000,timezone.utc).strftime('%m-%d %H:%M:%S'),reported_magnitude=rounded(r['values']['magnitude']),magType=r['magType'],depth_km=rounded(r['values']['depth_km'])) for i,r in enumerate(observed)]
        need(sha(canonical(shown))==record['display_rows_SHA256'],'Displayed rows numeric identity differs')
        image=record['image'];path=DATA/image['name'];need(path.resolve().is_relative_to(DATA.resolve()) and image['name'].startswith(record['split']+'/media/'),'Image path/split escape')
        raw=read(path,(image['bytes'],image['SHA256']));png_identity(raw);need(image['SHA256'] not in pnghashes,'Media duplicate');pnghashes.add(image['SHA256']);bytes_png+=len(raw)
        need(record['display_rows_SHA256'] not in displayhashes,'Duplicate displayed table');displayhashes.add(record['display_rows_SHA256'])
        depth=[Decimal(r['depth_km']) for r in shown];mag=[Decimal(r['reported_magnitude']) for r in shown]
        answers=[shown[0]['reported_magnitude'],shown[0]['depth_km'],rounded(depth[-1]-depth[0]),' '.join(r['label'] for r,v in zip(shown,depth) if v==max(depth)),' '.join(r['label'] for r,v in zip(shown,mag) if v==max(mag))]
        for i,(kind,text) in enumerate(expected_questions.items()):
            qid=record['id']+':'+str(i);labelsplit,label=alllabels[qid];insplit,inp=allinputs[qid]
            need(labelsplit==insplit==record['split'] and label['data_status']=='QUALIFIED' and label['kind']==kind and label['question']==text and label['answer']==answers[i]
                and label['answer_type']==('decimal2' if i<3 else 'labels') and label['display_rows_SHA256']==record['display_rows_SHA256'] and label['single_event_window']==(len(shown)==1)
                and label['chart_id']==record['id'] and label['group']==record['group'],'Displayed-value gold/label contract differs')
            need(set(inp)=={'id','chart_group','source_group','messages','images'} and inp['chart_group']==record['id'] and inp['source_group']==record['group']
                and inp['messages']==[dict(role='user',content='<image>\n'+text)] and inp['images']==[image['name']],'Input contains answer or different prompt/image binding')
            kinds[kind]+=1
        if record['split'] in visual:visual[record['split']].append(dict(chart_order_sha=sha(record['id'].encode()),single=len(shown)==1,path=path.as_posix(),image_SHA256=image['SHA256'],source_event_count=len(shown)))
    need(bytes_png==m['actual_PNG_bytes']==worker['actual_PNG_bytes']==548161 and len(alllabels)==len(allinputs)==80,'Full output/media denominator differs')
    need(m['renderer']['versions']==plan['renderer_versions'] and m['renderer']['font_SHA256']==plan['font_SHA256'] and m['renderer']['pixels']==[512,512] and m['renderer']['dpi']==100,'Renderer identities differ')
    # The protected qualified overlay font was denied in the first audit. No alternate read.
    # Frozen manifest font identity was checked above; font bytes remain NOT_READ in this audit.
    read(DATA/'CREDITS.json',( (DATA/'CREDITS.json').stat().st_size,m['credits_SHA256']))
    progress=[json.loads(line) for line in read(RUN/'progress.jsonl').splitlines()]
    need(progress[0]['phase']=='split_frozen' and [x['phase'] for x in progress[1:]]==['chart_terminal']*16,'Split not persisted before QA/chart progress')
    need(progress[0]['planned_charts']==16 and progress[0]['planned_QA']==80 and progress[-1]['completed_chart_slots']==16 and progress[-1]['completed_QA_slots']==80,'Progress full denominator differs')
    for split in visual:
        selected=[]
        for single in [True,False]:
            candidates=sorted([x for x in visual[split] if x['single']==single],key=lambda x:x['chart_order_sha'])
            if candidates:selected.append(candidates[0])
        visual[split]=selected
    (OUT/'visual-train-dev-selection-private.json').write_bytes(canonical(visual))
    private=dict(source_slots_checked=50,closure_edges_checked=len(expected_edges),all_source_event_coverage_exactly_once=True,no_alias_region_neighbour_crosssplit=True,
        all_labels_recomputed=True,answer_free_model_inputs=True,all_png_passive_headers_and_chunkCRCs=True,pixels_not_decoded=True,visual_selection=visual)
    (OUT/'audit-details-private.json').write_bytes(canonical(private))
    result=dict(passed=True,state='OFFLINE_DERIVED_DATA_METADATA_AND_NUMERIC_AUDIT_PASS_NOT_NATIVE_OR_EFFECT',original_request=REQUEST,publication=PUBLICATION,
        original_source_body_SHA256=plan['original_source_body_SHA256'],original_numeric_slots=50,original_producer_failed_preserved=True,
        original_body_acquisition_GETs=1,new_GETs=0,model_calls=0,actual_charts=16,planned_charts=16,failed_charts=0,planned_QA=80,actual_inputs=80,retained_QA_slots=80,
        split_QA=splitcounts,split_group_counts=m['split_group_counts'],source_alias_groups=50,source_geographic_groups=12,closed_source_groups=12,closure_edges=len(expected_edges),
        chart_singleton_count=singletons['True'],chart_multievent_count=singletons['False'],split_charts=dict(split_charts),split_singleton_charts={s:split_singletons[s] for s in ['train','dev','blind']},single_event_QA=singletons['True']*5,single_event_QA_fraction=singletons['True']/16,split_single_event_QA={s:split_singletons[s]*5 for s in ['train','dev','blind']},QA_per_kind=dict(kinds),duplicate_source_slots=0,crosssplit_event_or_group_overlap=0,
        duplicate_media=0,all_gold_derived_from_displayed_Decimal=True,all_inputs_answer_free=True,PNG_bytes=bytes_png,PNG_size=[512,512],passive_PNG_CRC_identity_verified=True,pixel_decode=False,
        renderer=m['renderer']['versions'],font_SHA256=plan['font_SHA256'],whole_epoch_seconds=120,actual_job_wall_seconds=job['wall_seconds'],CPU=2,
        actual_peak_job_commit_bytes=job['actual_peak_job_commit_bytes'],OS_sampled_peak_working_set_bytes=job['observed_OS_PeakWorkingSetSize_bytes'],
        renderer_or_build_rerun=False,first_audit_font_ACL_failure_preserved=True,font_bytes_independently_read=False,font_identity_basis='Frozen manifest/plan match; producer qualified preflight, not a fresh font read',blind_review='CODE_ONLY_LABEL_CHECK_NO_HUMAN_OR_IMAGE_VIEW',visual_readability='ROOT_BOUNDED_TWO_IMAGES_PASS_ONLY_ALL_OTHER_IMAGES_NOT_VISUALLY_REVIEWED',root_visual_review=dict(images=2,train_events=1,dev_events=4,pass_claim_from_root=True,reviewer='root view_image',source_manifest_PENDING_unchanged=True,media=[dict(path='train/media/000.png',SHA256=next(c['image']['SHA256'] for c in m['charts'] if c['image']['name']=='train/media/000.png')),dict(path='dev/media/009.png',SHA256=next(c['image']['SHA256'] for c in m['charts'] if c['image']['name']=='dev/media/009.png'))]),native_processor='NOT_RUN',
        source_window_completeness='UNKNOWN_LIMIT50_REACHED',mixed_magType_physical_comparison=False,domain_capability='NOT_EVALUATED',Dataset_GO=False,training_GO=False,GPU_GO=False,
        original_artifacts={name:dict(bytes=p[0],SHA256=p[1]) for name,p in PINS.items()},dataset_manifest_SHA256=sha(mraw),split_plan_SHA256=sha(sraw),split_outputs=files,
        reader_source_SHA256=sha(Path(__file__).read_bytes()),audit_details_SHA256=sha((OUT/'audit-details-private.json').read_bytes()),visual_selection_SHA256=sha((OUT/'visual-train-dev-selection-private.json').read_bytes()))
    (OUT/'result.json').write_bytes(canonical(result));print(json.dumps(result,sort_keys=True));return result

if __name__=='__main__':audit()
