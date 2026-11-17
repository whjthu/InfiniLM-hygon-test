"""Independent stdlib metadata/math audit. No producer imports or blind gold/media opens."""
import argparse
from collections import Counter
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re
import stat
import struct
import time
import zlib

SOURCE=Path(__file__).resolve().parent
G=SOURCE.parents[2]
BASE=G/'code/bw1000/training-worldbank-macro-chart-v1'
PLAN_SHA='c49d86cb991e9b3d10d08a0b953e63d44349f4e76fc6f16d1f855b676c97f2ec'
BASE_MF='93ef8719b59659e460b40bda8549bd2fd681e57eaa2cea6c9104d83a5e6034e6'


def require(value,why):
    if not value:raise ValueError(why)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()


def jload(raw,decimal=False):
    def pairs(values):
        require(len(values)==len({k for k,v in values}),'Duplicate JSON member');return dict(values)
    def constant(value):raise ValueError('Nonfinite JSON constant')
    return json.loads(raw,parse_float=Decimal if decimal else float,parse_constant=constant,object_pairs_hook=pairs)


class Audit:
    def __init__(self,root):self.root=root.resolve();self.end=time.monotonic()+60;self.bytes=0;self.pins={}
    def path(self,name):
        p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path')
        path=self.root.joinpath(*p.parts)
        for part in (path,*path.parents):
            if part==self.root.parent:break
            require(not part.is_symlink() and not getattr(part,'is_junction',lambda:False)(),'Symlink/junction forbidden')
        require(path.resolve().is_relative_to(self.root),'Path escaped run');return path
    def read(self,name,cap,expected=None):
        require(time.monotonic()<self.end,'Reader60s budget exhausted')
        path=self.path(name);before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'Not bounded regular file')
        raw=path.read_bytes();after=path.stat()
        require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino) and len(raw)==before.st_size,'File changed while reading')
        pin=dict(bytes=len(raw),SHA256=digest(raw));require(expected is None or pin['SHA256']==expected,'Original pin differs: '+name)
        self.bytes+=len(raw);require(self.bytes<=40*1024**2,'Reader total40MiB cap exceeded');self.pins[name]=pin;return raw
    def stat_only(self,name,declared):
        st=self.path(name).stat();require(stat.S_ISREG(st.st_mode) and st.st_size==declared['bytes'],'Unopened blind file stat differs')
        return dict(bytes=st.st_size,declared_SHA256=declared['SHA256'],bytes_SHA256_verified=False,opened=False)


def rounded(value):
    number=Decimal(value).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return format(number.copy_abs() if number==0 else number,'.2f')


def png_header(raw):
    require(raw[:8]==b'\x89PNG\r\n\x1a\n','Not PNG');offset=8;ihdr=None;software=None;dpi=None;end=False
    while offset<len(raw):
        require(offset+12<=len(raw),'Truncated PNG chunk');size=struct.unpack('>I',raw[offset:offset+4])[0]
        kind=raw[offset+4:offset+8];data=raw[offset+8:offset+8+size]
        require(len(data)==size and offset+12+size<=len(raw),'Truncated PNG data')
        require(zlib.crc32(kind+data)&0xffffffff==struct.unpack('>I',raw[offset+8+size:offset+12+size])[0],'PNG chunk CRC mismatch')
        if kind==b'IHDR':ihdr=struct.unpack('>IIBBBBB',data)
        if kind==b'tEXt' and data.startswith(b'Software\0'):software=data[9:].decode('latin1')
        if kind==b'pHYs':dpi=struct.unpack('>IIB',data)
        offset+=size+12
        if kind==b'IEND':end=True;break
    require(end and offset==len(raw) and ihdr[:2]==(512,512) and software=='training-worldbank-macro-chart-v1' and dpi==(3937,3937,1),'PNG identity/dimensions/dpi/Software differs')
    return dict(width=512,height=512,pixels=262144,dpi_metadata=100,Software=software,pixel_decode=False)


def raw_facts(payload,group,indicators):
    require(isinstance(payload,list) and len(payload)==2,'Bad API array');header,rows=payload
    require(header['page']==header['pages']==1 and int(header['total'])==len(rows)==168 and str(header['sourceid'])=='2','Source/header denominator')
    cells={}
    for row in rows:
        key=(row['countryiso3code'],row['indicator']['id'],int(row['date']))
        require(key[0] in group['countries'] and key[1] in indicators and key[2] in group['years'] and key not in cells,'Facts group/duplicate')
        require(row['unit']=='' and row['indicator']['value']==indicators[key[1]]['name'],'Unit/name mismatch')
        value=row['value'];require(value is None or (type(value) in (int,Decimal) and Decimal(value).is_finite() and abs(value)<=1000000),'Nonfinite/non-numeric source value')
        cells[key]=value
    require(len(cells)==168,'Missing fact slots');return cells,header['lastupdated']


def expected_answers(values,years):
    v=[Decimal(rounded(x)) for x in values]
    return dict(**{'lookup:'+str(n):rounded(v[k]) for n,k in enumerate((0,3,6))},
        **{'change:'+str(n):rounded(v[b]-v[a]) for n,(a,b) in enumerate(((0,3),(3,6),(0,6)))},
        **{'extrema:0':rounded(max(v)),'extrema:1':rounded(min(v)),'extrema:2':str(years[v.index(max(v))])})


def run(args):
    output=args.output;require(not output.exists(),'Reader output already exists')
    require(digest((BASE/'PLAN.json').read_bytes())==PLAN_SHA and digest((BASE/'MANIFEST.json').read_bytes())==BASE_MF,'Frozen base source metadata changed')
    plan=jload((BASE/'PLAN.json').read_bytes());audit=Audit(args.run)
    intent=jload(audit.read('intent.json',65536,args.intent_sha));outcome=jload(audit.read('outcome.json',131072,args.outcome_sha))
    require(intent['publication']==args.publication and intent['source_manifest_SHA256']==args.source_manifest_sha,'Actual published scope differs')
    epoch=intent['epoch'];deadline=intent['deadline'];require(type(epoch) in (float,int) and type(deadline) in (float,int) and math.isfinite(epoch) and math.isfinite(deadline) and deadline-epoch==300,'Original epoch differs')
    require(outcome['request']==intent['request'] and outcome['epoch']==epoch and outcome['deadline']==deadline and outcome['known_terminal'] is True,'Original terminal context differs')
    job=outcome['job'];require(job['WindowsJobObject_limits_verified'] is True and job['worker_CPU_count']==2 and job['job_CPU_affinity_verified'] is True
        and job['worker_CPU_affinity_verified_before_gate'] is True and job['worker_affinity_mask'].bit_count()==2
        and job['process_commit_limit_bytes']==job['job_commit_limit_bytes']==2147483648 and job['deadline']==deadline,'Actual Job limits differ')
    worker=None
    if args.worker_sha!='MISSING':worker=jload(audit.read('worker-result.json',262144,args.worker_sha))
    else:require(not audit.path('worker-result.json').exists(),'Worker unexpectedly exists')
    result=dict(reader_passed=True,known_terminal=True,planned_facts=504,planned_charts=72,planned_QA=648,actual_Job=job,
        producer_passed=bool(worker and worker.get('passed') is True),original_epoch=epoch,original_deadline=deadline,
        blind_media_or_gold_opened=False,Dataset_GO=False,training_GO=False,GPU_GO=False,visual_readability='UNKNOWN_PENDING_FIXED_SIX_CHART_REVIEW')
    selection=[]
    if job['returncode']!=0 or job['timed_out'] or not result['producer_passed']:
        result.update(state='CONSUMED_ORIGINAL_FAILURE_ALL_PLANNED_SLOTS_RETAINED',actual_attempted_GETs=worker.get('attempted_GETs') if worker else None,
            actual_completed_GETs=worker.get('completed_GETs') if worker else None,actual_inputs=worker.get('actual_inputs') if worker else 0)
    else:
        require(worker['epoch']==epoch and worker['deadline']==deadline and worker['attempted_GETs']==worker['completed_GETs']==7,'Worker scope differs')
        manifest=jload(audit.read('derived/manifest-private.json',2097152,worker['manifest_SHA256']))
        require((manifest['planned_facts'],manifest['planned_charts'],manifest['planned_QA'],manifest['label_rows'])==(504,72,648,648),'Manifest full denominator differs')
        require(manifest['renderer']['versions']==plan['renderer_versions'] and re.fullmatch('[0-9a-f]{64}',manifest['renderer']['font_SHA256']) is not None,'Renderer identity missing')
        raw={}
        for name,endpoint in plan['endpoints'].items():
            data=audit.read('raw/'+name+'.json',endpoint['cap_bytes'],manifest['source_JSON_SHA256'][name]);receipt=worker['http'][name]
            require(receipt['URL']==endpoint['URL'] and receipt['status']==200 and receipt['bytes']==len(data) and receipt['SHA256']==digest(data),'Actual HTTP identity differs')
            if name!='facts_blind':raw[name]=jload(data,True)
        require(sum(worker['http'][n]['bytes'] for n in plan['endpoints'])<=6946816,'HTTP total cap')
        credits=jload(audit.read('derived/CREDITS-private.json',262144))
        require(credits['license']=='CC BY 4.0' and credits['license_URL']=='https://creativecommons.org/licenses/by/4.0/' and credits['endorsement'] is False,'Credits contract')
        for n,e in plan['endpoints'].items():
            if 'indicator' not in e:continue
            m=raw[n][1][0];require(m['id']==e['indicator'] and m['unit']=='' and m['source']['id']=='2' and m['name']==plan['indicators'][m['id']]['name'],'Metadata source/meaning')
            require(credits['metadata'][m['id']]['sourceOrganization']==m['sourceOrganization'] and credits['metadata'][m['id']]['sourceNote']==m['sourceNote'],'Original attribution/source-note lost')
        charts=manifest['charts'];require(len(charts)==72 and len({c['id'] for c in charts})==72,'Chart identity/denominator')
        declared_hashes=[];declared_vectors=[]
        for c in charts:
            g=plan['splits'][c['split']]
            require(c['country_group'] in g['countries'] and c['indicator'] in plan['indicators'] and c['years']==g['years'],'Declared split/source grouping')
            require(c['id']=='wdi:'+c['country_group']+':'+c['indicator']+':'+str(g['years'][0])+'-'+str(g['years'][-1]),'Chart source ID differs')
            if c['status']=='QUALIFIED':
                require(c['image']['name'].startswith(c['split']+'/media/'),'Declared image crosses split')
                require(c['source_group_SHA256']==digest(canonical([c['country_group'],c['indicator'],c['years']])),'Source group identity')
                declared_hashes.append(c['image']['SHA256']);declared_vectors.append(c['numeric_vector_SHA256'])
        require(len(declared_hashes)==len(set(declared_hashes)) and len(declared_vectors)==len(set(declared_vectors)),'Declared duplicate media/value vector')
        country_sets=[set(plan['splits'][s]['countries']) for s in ('train','dev','blind')];year_sets=[set(plan['splits'][s]['years']) for s in ('train','dev','blind')]
        require(sum(map(len,country_sets))==len(set.union(*country_sets))==24 and sum(map(len,year_sets))==len(set.union(*year_sets))==21,'Country/year leakage')
        png_seen=set();label_checked=media_checked=0;qualified_inputs=0;missing_source=0
        for split in ('train','dev'):
            group=plan['splits'][split];cells,vintage=raw_facts(raw['facts_'+split],group,plan['indicators']);require(vintage==manifest['source_vintage'],'Snapshot vintage drift')
            labels_raw=audit.read('derived/'+split+'/labels-private.jsonl',2097152,manifest['outputs'][split+'/labels-private.jsonl']['SHA256'])
            inputs_raw=audit.read('derived/'+split+'/inputs.jsonl',2097152,manifest['outputs'][split+'/inputs.jsonl']['SHA256'])
            labels=[jload(x) for x in labels_raw.splitlines()];inputs=[jload(x) for x in inputs_raw.splitlines()]
            require(len(labels)==216 and len({x['id'] for x in labels})==216 and len({x['id'] for x in inputs})==len(inputs),'Split input/gold denominator or duplicate')
            by_id={x['id']:x for x in labels};iby_id={x['id']:x for x in inputs};expected_ids=set();splitcharts=[c for c in charts if c['split']==split]
            require(len(splitcharts)==24,'Charts per split')
            for c in splitcharts:
                country=c['country_group'];indicator=c['indicator'];years=c['years']
                require(country in group['countries'] and indicator in plan['indicators'] and years==group['years'],'Chart source grouping')
                values=[cells[country,indicator,y] for y in years]
                answers=expected_answers(values,years) if all(x is not None for x in values) else None
                if answers is None:missing_source+=1;require(c['status']=='MISSING_SOURCE_VALUES','Missing values silently filled')
                if c['status']=='QUALIFIED':
                    require(answers is not None,'Qualified missing source chart');image=c['image']
                    require(c['numeric_vector_SHA256']==digest(canonical([rounded(x) for x in values])),'Displayed vector identity differs')
                    require(image['name'].startswith(split+'/media/') and image['bytes']<=393216,'Image namespace/cap')
                    data=audit.read('derived/'+image['name'],393216,image['SHA256']);require(len(data)==image['bytes'] and digest(data) not in png_seen,'Image duplicate/size')
                    png_seen.add(digest(data));png_header(data);media_checked+=1
                for kind in ('lookup','change','extrema'):
                    for i in range(3):
                        key=kind+':'+str(i);identifier=c['id']+':'+key;expected_ids.add(identifier);label=by_id[identifier]
                        require(label['chart_group']==c['id'] and label['country_group']==country and label['kind']==kind,'Label group mismatch')
                        if c['status']=='QUALIFIED':
                            require(label['data_status']=='QUALIFIED' and label['answer']==answers[key] and label['answer_type']==('year' if key=='extrema:2' else 'decimal2'),'Label differs from displayed Decimal values')
                            question=label['question']
                            if kind=='lookup':require(f'displayed value for {years[(0,3,6)[i]]}' in question,'Lookup year/stem differs')
                            elif kind=='change':
                                a,b=((0,3),(3,6),(0,6))[i]
                                require(f'from {years[a]} to {years[b]}' in question and 'later value minus the earlier value' in question and 'percentage points' in question,'Change direction/unit differs')
                            else:require(('maximum displayed value','minimum displayed value','maximum displayed value reached')[i] in question and (i!=2 or 'earliest year' in question),'Extrema/tie question differs')
                            item=iby_id[identifier];require(item['images']==[c['image']['name']] and item['messages']==[dict(role='user',content='<image>\n'+label['question'])],'Input/gold leak or prompt mismatch')
                            require(not set(item)&{'answer','label','labels','gold','response'} and 'FINAL:' in label['question'],'Answer-free schema')
                            qualified_inputs+=1
                        else:require(label['data_status']==c['status'] and label['answer'] is None and identifier not in iby_id,'Failed slot became model input/gold')
                        label_checked+=1
            require(expected_ids==set(by_id) and set(iby_id)=={x['id'] for x in labels if x['data_status']=='QUALIFIED'},'Split extra/missing membership')
            # Preselect by ID hash only, all slots eligible; failed chosen slot is retained, never backfilled.
            for indicator in sorted(plan['indicators']):
                chosen=min((c for c in splitcharts if c['indicator']==indicator),key=lambda c:hashlib.sha256(('WDI-visual-QA-v1:'+split+':'+c['id']).encode()).hexdigest())
                selection.append(dict(split=split,indicator=indicator,chart_id=chosen['id'],status=chosen['status'],image=chosen.get('image'),readability='NOT_YET_REVIEWED',blind=False))
        blind={}
        for name in ('inputs.jsonl','labels-private.jsonl'):
            key='blind/'+name;blind[key]=audit.stat_only('derived/'+key,manifest['outputs'][key])
        for c in charts:
            if c['split']=='blind' and c['status']=='QUALIFIED':blind[c['image']['name']]=audit.stat_only('derived/'+c['image']['name'],c['image'])
        result.update(state='TRAIN_DEV_SOURCE_LABEL_AND_PASSIVE_MEDIA_CHECKED_BLIND_UNOPENED',train_dev_label_slots_verified=label_checked,
            train_dev_native_input_records_verified=qualified_inputs,train_dev_media_headers_SHA_verified=media_checked,train_dev_missing_source_charts=missing_source,
            planned_blind_QA=216,blind_content_semantics_verified=False,blind_file_stat_only=blind,blind_raw_snapshot_SHA_only=True,
            visual_selection_slots=6,producer_actual_charts=manifest['actual_charts'],producer_actual_inputs=manifest['actual_inputs'],renderer=manifest['renderer'])
    result.update(total_actual_bytes_read=audit.bytes,original_files=audit.pins,source_base_publication='13f22571244ccbd6ebf5e8f2654108e76dcb12a5',no_HTTP=True,no_render=True,no_model=True)
    output.mkdir();(output/'result-private.json').write_bytes(canonical(result));(output/'visual-QA-selection-private.json').write_bytes(canonical(selection))
    print(json.dumps(dict(reader_passed=True,state=result['state'],planned_QA=648,blind_opened=False,result_SHA256=digest(canonical(result)),visual_selection_slots=len(selection))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for n in ('intent-sha','outcome-sha','worker-sha','publication'):p.add_argument('--'+n,required=True)
    p.add_argument('--source-manifest-sha',default=BASE_MF)
    a=p.parse_args()
    for n in ('intent_sha','outcome_sha'):require(re.fullmatch('[0-9a-f]{64}',getattr(a,n)) is not None,'Bad original SHA')
    require(a.worker_sha=='MISSING' or re.fullmatch('[0-9a-f]{64}',a.worker_sha) is not None,'Bad worker SHA')
    require(re.fullmatch('[0-9a-f]{40}',a.publication) is not None and re.fullmatch('[0-9a-f]{64}',a.source_manifest_sha) is not None,'Bad scope binding')
    run(a)
