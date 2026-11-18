"""Independent original source/closure/gold/PNG audit; no HTTP/render/model imports."""
import argparse,hashlib,json,math,struct,zlib
from datetime import datetime,timezone
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
D=Path(__file__).resolve().parent;G=D.parents[2]


def need(ok,why):
    if not ok:raise ValueError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def display(x):
    y=Decimal(x).quantize(Decimal('.01'),rounding=ROUND_HALF_UP);return format(Decimal('0') if y==0 else y,'.2f')


def main(out):
    need(not out.exists(),'EXCLUSIVE_NEW_AUDIT_OUTPUT');out.mkdir(mode=0o700);pins={}
    def read(path,pin=None,cap=2*1024**2):
        a=path.lstat();need(path.is_file() and not path.is_symlink() and 0<a.st_size<=cap,'BOUNDED_REGULAR_ORIGINAL')
        raw=path.read_bytes();b=path.lstat()
        need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'STABLE_ORIGINAL')
        p=dict(bytes=len(raw),SHA256=sha(raw));need(pin is None or all(p[k]==pin[k] for k in ['bytes','SHA256']),'EXACT_ORIGINAL_PIN')
        pins[str(path.relative_to(G))]=p;return raw
    binding=json.loads((D/'BINDING.json').read_bytes());src=G/binding['producer_source'];run=G/binding['original_namespace']
    mfraw=read(src/'MANIFEST.json',binding['producer_manifest']);mf=json.loads(mfraw)
    for name,pin in mf['files'].items():read(src/name,pin)
    need(sha((src/'CLAIM.json').read_bytes())==binding['producer_claim_SHA256'],'ACTUAL_ADOPTED_CLAIM')
    plan=json.loads((src/'PLAN.json').read_bytes());docs={n:json.loads(read(run/n,pin)) for n,pin in binding['original_files'].items()}
    intent=docs['intent.json'];outcome=docs['outcome.json'];worker=docs['worker-result.json'];job=outcome['job']
    for v in [intent,outcome]:need(v['request']==binding['request'] and v['publication']==binding['publication'] and v['epoch']==binding['epoch'] and v['deadline']==binding['epoch']+120 and v['source_manifest_SHA256']==binding['producer_manifest']['SHA256'],'ACTUAL_SINGLE120_CONTEXT')
    need(job['returncode']==0 and job['timed_out'] is False and outcome['known_terminal'] is True
         and job['WindowsJobObject_limits_verified'] is True and job['worker_CPU_count']==2
         and job['job_commit_limit_bytes']==job['process_commit_limit_bytes']==2147483648,'ACTUAL_JOB_CPU2_COMMIT2G')
    need(worker['passed'] is True and worker['actual_GETs']==1 and worker['actual_charts']==5 and worker['actual_QA']==25,'ACTUAL_PRODUCER_DENOMINATOR')
    raw=read(run/'source-GeoJSON-private.json',dict(bytes=worker['new_body_bytes'],SHA256=worker['source_body_SHA256']),1048576)
    source=json.loads(raw,parse_float=Decimal);facts=source['features'];meta=source['metadata'];need(source['type']=='FeatureCollection' and meta['status']==200 and len(facts)==worker['new_source_slots']<=200,'ACTUAL_SOURCE_FEATURE_DENOMINATOR')
    if 'count' in meta:need(meta['count']==len(facts),'OPTIONAL_REAL_COUNT')
    numeric=json.loads(read(run/'source-numerics-private.json'));rows=numeric['rows'];need(len(rows)==len(facts),'SOURCE_NUMERIC_ALL_ROWS')
    start=int(datetime.fromisoformat(plan['fixed_start_UTC']).timestamp()*1000);end=int(datetime.fromisoformat(plan['fixed_end_UTC']).timestamp()*1000)
    for f,r in zip(facts,rows):
        p=f['properties'];lon,lat,depth=f['geometry']['coordinates'];values=dict(magnitude=str(p['mag']),longitude=str(lon),latitude=str(lat),depth_km=str(depth))
        need(f['type']=='Feature' and f['geometry']['type']=='Point' and p['type']=='earthquake' and r['id']==f['id'] and r['values']==values,'ORIGINAL_REPORTED_VALUES_EXACT_DECIMAL')
        need(Decimal('4.5')<=Decimal(p['mag'])<=20 and -180<=lon<=180 and -90<=lat<=90 and -100<=depth<=1000
             and type(p['time']) is int and start<=p['time']<=end and p['updated']>=p['time'],'NUMERIC_UNITS_TIME_RANGE')
        expected_region='lat30_'+str(min(5,int((lat+90)//30)))+'_lon60_'+str(min(5,int((lon+180)//60)))
        need(r['time_UTC_epoch_ms']==p['time'] and r['updated_UTC_epoch_ms']==p['updated'] and r['aliases']==[x for x in p['ids'].split(',') if x]
             and r['magType']==p['magType'] and r['source_geo_region_key']==expected_region and r['status']=='QUALIFIED_SOURCE_NUMERIC_RECORD'
             and r['errors']==r['null_fields']==[],'EXACT_ALL_SOURCE_IDENTITIES')
    need(len({r['id'] for r in rows})==len(rows) and [r['time_UTC_epoch_ms'] for r in rows]==sorted(r['time_UTC_epoch_ms'] for r in rows),'SOURCE_UNIQUE_ASCENDING')
    old=json.loads(read(G/plan['old_source_path'],plan['old_source_pin']))['rows'];need(len(old)==50,'OLD_ALL50_NO_GOLD')
    allrows=old+rows;parent=list(range(len(allrows)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(allrows):
        for j in range(i):
            b=allrows[j];close=set(a['aliases'])&set(b['aliases']) or a['source_geo_region_key']==b['source_geo_region_key']
            if not close and abs(a['time_UTC_epoch_ms']-b['time_UTC_epoch_ms'])<=72*3600000:
                x,y=math.radians(float(a['values']['latitude'])),math.radians(float(b['values']['latitude']));dlon=math.radians(float(b['values']['longitude'])-float(a['values']['longitude']))
                h=math.sin((y-x)/2)**2+math.cos(x)*math.cos(y)*math.sin(dlon/2)**2
                close=6371.0088*2*math.asin(min(1,math.sqrt(max(0,h))))<=500
            if close:
                x,y=root(i),root(j)
                if x!=y:parent[max(x,y)]=min(x,y)
    oldroots={root(i) for i in range(50)};groups={};excluded=[]
    for i,r in enumerate(rows,50):
        if root(i) in oldroots:excluded.append(i-50)
        else:groups.setdefault(root(i),[]).append(r)
    ordered=[]
    for members in groups.values():
        members=sorted(members,key=lambda r:(r['time_UTC_epoch_ms'],r['id']));gid=sha(canonical(sorted(r['id'] for r in members)))
        ordered.append(dict(group=gid,order=sha((plan['seed']+gid).encode()),windows=[members[i:i+5] for i in range(0,len(members),5)]))
    ordered.sort(key=lambda x:(x['order'],x['group']));chosen=[];round_index=0
    while len(chosen)<5 and any(round_index<len(g['windows']) for g in ordered):
        for group in ordered:
            if round_index<len(group['windows']):chosen.append((group['group'],round_index,group['windows'][round_index]))
            if len(chosen)==5:break
        round_index+=1
    need(len(chosen)==5 and len({g for g,k,window in chosen})>=3,'PRE_REGISTERED_FIVE_THREE')
    selected=json.loads(read(run/'closure-selection-private.json'));manifest_raw=read(run/'dataset/manifest-private.json');manifest=json.loads(manifest_raw)
    need(sha(manifest_raw)==worker['manifest_SHA256'] and selected['excluded_new_slots']==excluded and selected['eligible_groups']==len(groups),'INDEPENDENT_CLOSURE_COUNT')
    declaration_raw=read(run/'dataset/declaration-private.json');declaration=json.loads(declaration_raw);need(sha(declaration_raw)==manifest['declaration_SHA256'],'DURABLE_PRE_QA_DECLARATION')
    input_pin=manifest['outputs']['confirmation/inputs.jsonl'];label_pin=manifest['outputs']['confirmation/labels-private.jsonl']
    inputs=[json.loads(x) for x in read(run/'dataset/confirmation/inputs.jsonl',input_pin).splitlines()]
    labels=[json.loads(x) for x in read(run/'dataset/confirmation/labels-private.jsonl',label_pin).splitlines()]
    need(len(inputs)==len(labels)==25 and len({x['id'] for x in inputs})==25,'FULL25_SEPARATE_LABELS')
    png_hashes=[];counts=[];chartpins=[]
    for n,(gid,k,window) in enumerate(chosen):
        cid='usgs-confirm:'+gid+':'+str(k);actual=selected['charts'][n];decl=declaration['charts'][n]
        need(actual['id']==decl['id']==cid and actual['group']==decl['group']==gid and actual['rows']==window and decl['event_ids']==[r['id'] for r in window],'INDEPENDENT_FIXED_WINDOW_SELECTION')
        image='confirmation/media/'+str(n).zfill(3)+'.png';png=read(run/'dataset'/image,manifest['media'][image],1048576);png_hashes.append(sha(png));counts.append(len(window))
        need(png[:8]==b'\x89PNG\r\n\x1a\n','PNG_SIGNATURE');offset=8;end_seen=False;dimensions=None
        while offset<len(png):
            length=struct.unpack('>I',png[offset:offset+4])[0];kind=png[offset+4:offset+8];body=png[offset+8:offset+8+length];checksum=struct.unpack('>I',png[offset+8+length:offset+12+length])[0]
            need((zlib.crc32(kind+body)&0xffffffff)==checksum,'PNG_FULL_CHUNK_CRC')
            if kind==b'IHDR':dimensions=struct.unpack('>II',body[:8])
            if kind==b'IEND':need(length==0 and offset+12==len(png),'PNG_EXACT_END');end_seen=True
            offset+=length+12
        need(offset==len(png) and end_seen and dimensions==(512,512),'PNG_COMPLETE_GEOMETRY')
        depth=[Decimal(display(r['values']['depth_km'])) for r in window];mag=[Decimal(display(r['values']['magnitude'])) for r in window]
        expected=[display(window[0]['values']['magnitude']),display(window[0]['values']['depth_km']),display(depth[-1]-depth[0]),
            ' '.join('E'+str(i+1).zfill(2) for i,x in enumerate(depth) if x==max(depth)),
            ' '.join('E'+str(i+1).zfill(2) for i,x in enumerate(mag) if x==max(mag))]
        kinds=['lookup_reported_magnitude','lookup_depth','depth_change','maximum_depth','maximum_reported_number']
        for j,answer in enumerate(expected):
            q=labels[n*5+j];inp=inputs[n*5+j]
            need(q['id']==inp['id']==cid+':'+str(j) and q['chart_id']==inp['chart_group']==cid and q['group']==inp['source_group']==gid
                 and q['kind']==kinds[j] and q['data_status']=='QUALIFIED' and q['answer']==answer,'ALL25_INDEPENDENT_DISPLAY_GOLD')
            need(set(inp)=={'id','chart_group','source_group','messages','images'} and inp['messages']==[dict(role='user',content='<image>\n'+q['question'])]
                 and inp['images']==[image],'ANSWER_FREE_USER_ONLY_NATIVE_INTERFACE')
        chartpins.append(dict(path=str(run/'dataset'/image),**manifest['media'][image]))
    need(len(set(png_hashes))==5 and sum(counts)==len({r['id'] for g,k,w in chosen for r in w})==manifest['selected_events'],'NEW_EVENT_AND_MEDIA_UNIQUENESS')
    need(manifest['renderer']['font_SHA256']==plan['font_SHA256']=='3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22' and manifest['renderer']['pixels']==[512,512],'ACTUAL_QUALIFIED_RENDERER_FONT_BINDING')
    summary=dict(passed=True,state='CONFIRMATION_DATA_INDEPENDENTLY_QUALIFIED_NATIVE_PENDING',request=binding['request'],publication=binding['publication'],original_GETs=1,new_audit_GETs=0,
        actual_new_source_slots=len(rows),old_source_slots=50,excluded_connected_to_old=len(excluded),eligible_new_groups=len(groups),selected_groups=len({g for g,k,w in chosen}),
        selected_event_count=sum(counts),events_per_chart=counts,single_event_charts=sum(x==1 for x in counts),single_event_QA=5*sum(x==1 for x in counts),charts=5,QA=25,
        numeric_source_slots_verified=len(rows),gold_answers_verified=25,user_only_inputs=25,PNG_full_CRC_verified=5,PNG_geometry=[512,512],native_processor='NOT_RUN',
        visual_readability='PENDING_ROOT_REVIEW',font_body_independently_read=False,font_basis='Producer qualified renderer/preflight and exactmanifest/planSHA; no new fontbody read',
        duplicate_selected_events=0,duplicate_new_PNG=0,old_new_frozen_closure_overlap=0,absolute_no_leakage_proven=False,whole_catalog_completeness='UNKNOWN_FIRST200_FILTERED_QUERY',
        renderer_or_build_rerun=False,model_calls=0,Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False,prior_dataset_manifest_permission_failure_preserved=True,
        dataset_manifest=dict(bytes=len(manifest_raw),SHA256=sha(manifest_raw)),private_inputs=input_pin,private_labels=label_pin)
    (out/'result.json').write_bytes(canonical(summary));(out/'original-file-pins-private.json').write_bytes(canonical(pins));(out/'visual-review-paths-private.json').write_bytes(canonical(chartpins))
    print(json.dumps(summary,allow_nan=False));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
