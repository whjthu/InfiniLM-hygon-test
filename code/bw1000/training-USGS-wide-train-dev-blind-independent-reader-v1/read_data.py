"""Single bounded independent saved-data audit; output contains no questions/answers."""
import hashlib
import json
import os
import stat
import struct
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from independent_math import require, canonical, digest, decode, source_rows, reconstruct, chart_targets

D = Path(__file__).resolve().parent
G = D.parents[2]


class Reader:
    def __init__(self, epoch, deadline, cap):
        self.end = time.monotonic() + deadline-time.time()
        self.deadline, self.total, self.cap, self.pins, self.cache = deadline, 0, cap, {}, {}
    def check(self):
        require(min(self.end-time.monotonic(),self.deadline-time.time())>8, 'ORIGINAL120_READER_DEADLINE')
    def take(self, path, pin=None, limit=2*1024**2):
        self.check();path=Path(path);key=path.relative_to(G).as_posix()
        if key not in self.cache:
            require(not path.is_symlink() and path.resolve().is_relative_to(G.resolve()),'READ_PATH_SCOPE')
            before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_size<=limit
                and self.total+before.st_size<=self.cap,'READ_BYTE_CAP')
            def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            with path.open('rb') as f:
                require(identity(before)==identity(os.fstat(f.fileno())),'OPENED_FILE_IDENTITY')
                raw=f.read(before.st_size+1);self.total+=len(raw)
                require(len(raw)==before.st_size and identity(before)==identity(os.fstat(f.fileno()))==identity(path.stat()),'STABLE_ORIGINAL_FILE')
            self.pins[key]=dict(bytes=len(raw),SHA256=digest(raw));self.cache[key]=raw
        raw=self.cache[key]
        if pin is not None:
            require(len(raw)==pin['bytes'] and digest(raw)==pin['SHA256'],'EXACT_ORIGINAL_PIN')
        return raw
    def js(self,path,pin=None,limit=2*1024**2):return decode(self.take(path,pin,limit))


def png_check(raw, software):
    require(raw[:8]==b'\x89PNG\r\n\x1a\n','PNG_SIGNATURE')
    pos=8;types=[];idat=[];software_seen=False
    while pos<len(raw):
        require(pos+12<=len(raw),'PNG_CHUNK_HEADER')
        n=struct.unpack('>I',raw[pos:pos+4])[0];typ=raw[pos+4:pos+8]
        require(n<=1048576 and pos+12+n<=len(raw),'PNG_CHUNK_BOUND')
        data=raw[pos+8:pos+8+n];crc=struct.unpack('>I',raw[pos+8+n:pos+12+n])[0]
        require(zlib.crc32(typ+data)&0xffffffff==crc,'PNG_CRC')
        types.append(typ)
        if typ==b'IHDR':require(len(types)==1 and struct.unpack('>IIBBBBB',data)==(512,512,8,6,0,0,0),'PNG512_RGBA8_NONINTERLACED')
        elif typ==b'IDAT':idat.append(data)
        elif typ==b'tEXt' and data.startswith(b'Software\x00'):
            require(data.split(b'\x00',1)[1].decode()==software,'PNG_SOFTWARE');software_seen=True
        elif typ==b'IEND':require(n==0 and pos+12==len(raw),'PNG_END')
        elif not (typ[0]&32):raise ValueError('UNKNOWN_PNG_CRITICAL_CHUNK')
        pos+=n+12
    require(types.count(b'IHDR')==types.count(b'IEND')==1 and types[-1]==b'IEND'
            and idat and software_seen,'PNG_COMPLETE')
    size=512*(512*4+1);decoder=zlib.decompressobj();pixels=decoder.decompress(b''.join(idat),size+1)
    require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data and len(pixels)==size
            and all(pixels[k] in range(5) for k in range(0,size,2049)),'PNG_DECODED_ROWS_BOUND')


def difficulty(charts):
    out={}
    for split in ('train','dev','blind','all'):
        selected=[c for c in charts if split=='all' or c['split']==split]
        counts=Counter();event_counts=Counter();constant_depth=constant_mag=single=0
        for c in selected:
            shown,_=chart_targets(c);event_counts[len(shown)]+=1;single+=len(shown)==1
            depths={Decimal(r['depth_km']) for r in shown};mags={Decimal(r['reported_magnitude']) for r in shown}
            constant_depth+=len(depths)==1;constant_mag+=len(mags)==1
            if len(depths)==1:
                counts['depth_change_constant_input']+=1
                counts['maximum_depth_all_tied']+=1
            if len(mags)==1:counts['maximum_reported_number_all_tied']+=1
        out[split]=dict(charts=len(selected),QA=5*len(selected),event_count_histogram={str(k):v for k,v in sorted(event_counts.items())},
            single_event_charts=single,multi_event_charts=len(selected)-single,constant_display_depth_charts=constant_depth,
            constant_display_magnitude_charts=constant_mag,input_degenerate_tasks_by_kind=dict(counts),
            input_degenerate_tasks_total=sum(counts.values()),single_event_structurally_degenerate_tasks=3*single,
            multi_event_QA=5*(len(selected)-single),single_event_QA=5*single,
            difficulty_is_not_measured_model_accuracy=True)
    return out


def main(epoch,deadline,outdir):
    start=time.monotonic();binding=decode((D/'BINDING.json').read_bytes());out=Path(outdir)
    from clock_ipc import validate_gate
    from job_guard import current_affinity
    r=Reader(epoch,deadline,binding['metadata_total_read_cap'])
    while not (out/'job-gate.json').exists():r.check();time.sleep(.02)
    validate_gate((out/'job-gate.json').read_bytes(),epoch,deadline,current_affinity()['process_mask'])
    result=dict(passed=False,planned_source_slots=500,planned_prior_slots=250,planned_charts=30,planned_QA=150,
        verified_charts=0,verified_QA=0,GETs=0,render_calls=0,model_calls=0,score_calls=0,
        Dataset_GO=False,training_GO=False,GPU_GO=False,Model_GO=False,visual_semantic_pixel_audit='NOT_PERFORMED',
        private_labels_read_for_data_validation=False,labels_read_for_data_integrity=False,model_predictions=0,scores=0,raw_question_or_answer_output=False,epoch=epoch,deadline=deadline)
    phase='source-and-original-context'
    try:
        s=G/binding['producer_source'];run=G/binding['original_namespace'];dataset=run/'dataset'
        mf=r.js(s/'MANIFEST.json',binding['producer_manifest'])
        for name,pin in mf['files'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'PRODUCER_SOURCE_PATH')
            r.take(s/name,pin)
        plan=r.js(s/'PLAN.json');protocol=r.js(s/'EVALUATION_PROTOCOL.json',plan['evaluation_protocol'])
        intent=r.js(run/'intent.json');terminal=r.js(run/'outcome.json');worker=r.js(run/'worker-result.json')
        require(intent['request']==binding['original_request'] and intent['publication']==binding['original_publication']
            and intent['source_manifest_SHA256']==binding['producer_manifest']['SHA256']
            and intent['claim_SHA256']==digest(r.take(s/'CLAIM.json')),'ORIGINAL_PUBLICATION_REQUEST_BINDING')
        require(all(terminal[k]==v for k,v in intent.items()) and terminal['known_terminal'] is True,'ORIGINAL_TERMINAL_CONTEXT')
        job=terminal['job'];require(job['returncode']==0 and job['timed_out'] is False
            and job['WindowsJobObject_limits_verified'] is True and job['worker_CPU_count']==2
            and job['process_commit_limit_bytes']==job['job_commit_limit_bytes']==2147483648,'ORIGINAL_JOB_TERMINAL')
        require(worker['passed'] is True and worker['epoch']==intent['epoch'] and worker['deadline']==intent['deadline']==intent['epoch']+120
                and worker['actual_GETs']==1 and worker['actual_charts']==30 and worker['actual_QA']==150,'ORIGINAL_WORKER_DENOMINATOR')
        manifest=r.js(run/binding['manifest']['name'],binding['manifest'])
        require(worker['manifest_SHA256']==binding['manifest']['SHA256'],'ORIGINAL_MANIFEST_LINK')
        raw=r.take(run/binding['body']['name'],binding['body']);phase='independent-source-numerics'
        fresh,meta=source_rows(raw,plan);saved=r.js(run/'source-numerics-private.json')
        require(saved['rows']==fresh and saved['actual_event_slots']==500 and saved['all_source_numeric_slots_qualified'] is True
            and saved['sample_limit_reached'] is True and saved['whole_catalog_window_complete']=='UNKNOWN_NOT_CLAIMED','INDEPENDENT500_NUMERIC_ROWS')
        require(saved['generated_epoch_ms']==meta['generated'] and saved['api_version']==meta['api']
            and saved['event_ids_ordered_SHA256']==digest(canonical([x['id'] for x in fresh])),'INDEPENDENT_SOURCE_ORDER_VINTAGE')
        cohort=r.js(run/'cohort-private.json')
        expected_cohort=dict(requested_start_UTC=plan['fixed_start_UTC'],requested_end_UTC=plan['fixed_end_UTC'],returned_event_count=500,
            sample_limit_reached=True,first_event_UTC_epoch_ms=fresh[0]['time_UTC_epoch_ms'],last_event_UTC_epoch_ms=fresh[-1]['time_UTC_epoch_ms'],
            first_event_UTC=datetime.fromtimestamp(fresh[0]['time_UTC_epoch_ms']/1000,timezone.utc).isoformat(),
            last_event_UTC=datetime.fromtimestamp(fresh[-1]['time_UTC_epoch_ms']/1000,timezone.utc).isoformat(),
            policy=plan['hit_cap_policy'],whole_month_catalog_complete=False)
        require(cohort==expected_cohort==worker['cohort'],'ACTUAL_FIRST500_NOT_FULL_MONTH')
        oldparts=[r.js(G/plan['prior_sources'][key]['path'],plan['prior_sources'][key]) for key in ('old50','new200')]
        require([x['actual_event_slots'] for x in oldparts]==[50,200] and all(x['all_source_numeric_slots_qualified'] is True for x in oldparts),'PRIOR250_PINS_SCHEMA')
        old=oldparts[0]['rows']+oldparts[1]['rows'];prior=r.js(G/plan['prior_sources']['closure']['path'],plan['prior_sources']['closure'])
        require(prior['old_slots']==50 and prior['new_slots']==200 and prior['passed'] is True,'PRIOR2724_SCHEMA')
        lower=int(datetime.fromisoformat(plan['fixed_start_UTC']).timestamp()*1000);upper=int(datetime.fromisoformat(plan['fixed_end_UTC']).timestamp()*1000)
        require(all(type(x['time_UTC_epoch_ms']) is int and not lower<=x['time_UTC_epoch_ms']<=upper for x in old),'PRIOR_WINDOW_SEPARATION')
        phase='independent-all-pairs-closure-split'
        selection=reconstruct(old,fresh,plan,prior['closure_edges'],r.check)
        require(r.js(run/'closure-selection-private.json')==selection,'INDEPENDENT_CLOSURE_SELECTION_ALL_FIELDS')
        declaration={k:v for k,v in selection.items() if k!='charts'}
        declaration['charts']=[dict(id=c['id'],group=c['group'],split=c['split'],event_ids=[x['id'] for x in c['rows']]) for c in selection['charts']]
        declaration_raw=r.take(dataset/'declaration-private.json')
        require(decode(declaration_raw)==declaration and digest(declaration_raw)==manifest['declaration_SHA256'],'DECLARED_GROUPS_BEFORE_TARGETS')
        phase='independent-all30-images-and150-targets'
        expected={s:{'inputs.jsonl':[],'labels-private.jsonl':[],'response-schemas-private.jsonl':[]} for s in ('train','dev','blind')}
        chart_records=[];media={};image_hashes=set();png_bytes=0
        for i,c in enumerate(selection['charts']):
            r.check();shown,targets=chart_targets(c);name=c['split']+'/media/'+f'{i:03d}.png'
            require(name in manifest['media'],'EXACT_MEDIA_NAME');pin=manifest['media'][name]
            png=r.take(dataset/name,pin,1048576);png_check(png,plan['Software_metadata'])
            require(pin==dict(bytes=len(png),SHA256=digest(png),width=512,height=512,name=name)
                and pin['SHA256'] not in image_hashes,'UNIQUE30_PNG_IDENTITY')
            image_hashes.add(pin['SHA256']);media[name]=pin;png_bytes+=len(png)
            chart_records.append(dict(id=c['id'],group=c['group'],split=c['split'],planned_QA=5,status='QUALIFIED',image=pin,
                display_rows_SHA256=digest(canonical(shown)),source_events=len(c['rows'])))
            for q in targets:
                expected[c['split']]['inputs.jsonl'].append(dict(id=q['id'],chart_group=c['id'],source_group=c['group'],
                    messages=[dict(role='user',content='<image>\n'+q['question']+protocol['prompt_suffix'])],images=[name]))
                expected[c['split']]['labels-private.jsonl'].append(dict(q,data_status='QUALIFIED',split=c['split']))
                expected[c['split']]['response-schemas-private.jsonl'].append(dict(id=q['id'],split=c['split'],kind=q['kind'],event_ids=[x['label'] for x in shown]))
            result['verified_charts']+=1
        require(manifest['charts']==chart_records and manifest['media']==media and manifest['PNG_bytes']==png_bytes<=10485760,'FULL30_MEDIA_MANIFEST')
        output_names={split+'/'+name for split,parts in expected.items() for name in parts}
        require(set(manifest['outputs'])==output_names,'EXACT_NINE_SPLIT_OUTPUTS')
        for split,parts in expected.items():
            for name,values in parts.items():
                key=split+'/'+name;pin=manifest['outputs'][key];content=r.take(dataset/key,pin)
                if name=='labels-private.jsonl':
                    result['private_labels_read_for_data_validation']=True;result['labels_read_for_data_integrity']=True
                rows=[decode(line) for line in content.splitlines()]
                require(rows==values and pin['rows']==len(rows)==50 and content==b''.join(canonical(v) for v in values),'INDEPENDENT_SPLIT_INPUT_TARGET_SCHEMA')
                if name=='labels-private.jsonl':result['verified_QA']+=50
        phase='credits-and-terminal-consistency'
        source=dict(cohort=cohort,evaluation_protocol_SHA256=plan['evaluation_protocol']['SHA256'],body_SHA256=digest(raw),api_version=meta['api'],generated_epoch_ms=meta['generated'],query_URL_SHA256=digest(plan['URL'].encode()))
        credits=dict(publisher='U.S. Geological Survey Earthquake Catalog',URL=plan['URL'],source_body_SHA256=digest(raw),generated_epoch_ms=meta['generated'],api_version=meta['api'],
            rights='USGS-produced numerical data U.S.publicdomain/free use with attribution; source networks retained. No worldwideCC0 or thirdparty image rights claim.',rights_URL='https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted',
            modifications='New512x512 local plots, displayedDecimal2 and frozen5QA kinds; not official USGS images, not energy/risk comparisons.',endorsement=False)
        require(r.js(dataset/'CREDITS.json')==credits and manifest['source']==source and manifest['prior_sources']==plan['prior_sources']
            and manifest['evaluation_protocol']==plan['evaluation_protocol'],'SOURCE_CREDITS_PROVENANCE')
        require(manifest['split_counts']==selection['split_counts'] and manifest['selected_groups']==selection['selected_group_count']
            and manifest['selected_events']==selection['selected_event_count'] and manifest['eligible_groups']==selection['eligible_groups']
            and manifest['new_source_slots']==500 and manifest['excluded_new_slots']==selection['excluded_count'],'MANIFEST_ALL_GROUP_DENOMINATORS')
        require(manifest['planned_charts']==manifest['actual_charts']==30 and manifest['planned_QA']==manifest['actual_inputs']==manifest['retained_QA']==150
            and manifest['strong_component_overlap']==0 and manifest['geographic_overlap_allowed'] is True and manifest['geographic_OOD'] is False
            and manifest['no_absolute_leakage_guarantee'] is True and manifest['native_processor']=='NOT_RUN'
            and all(manifest[k] is False for k in ('Dataset_GO','training_GO','GPU_GO','Model_GO')),'DATA_ONLY_LIMITS')
        renderer=manifest['renderer'];require(renderer['versions']==plan['renderer_versions'] and renderer['font_SHA256']==plan['font_SHA256']
            and renderer['pixels']==[512,512] and renderer['dpi']==100 and renderer['Software']==plan['Software_metadata'],'RECORDED_RENDERER_CONTRACT')
        result.update(passed=True,source_slots_verified=500,prior_history_edges_verified=2724,
            strong_edges=len(selection['closure_edges']),eligible_groups=selection['eligible_groups'],selected_groups=selection['selected_group_count'],
            selected_events=selection['selected_event_count'],split_counts=selection['split_counts'],input_difficulty=difficulty(selection['charts']),
            cohort=cohort,PNG_bytes=png_bytes,PNG_structure_and_identity_checked=30,
            private_labels_validated_without_model_or_scoring=True,geographic_OOD=False,no_absolute_leakage_guarantee=True)
    except BaseException as error:
        message=str(error);safe=(message if __import__('re').fullmatch('[A-Z0-9_]{1,100}',message) else None)
        result.update(error_type=type(error).__name__,safe_error_code=safe,error_code_SHA256=digest(message.encode()))
    result.update(phase=phase,pins=r.pins,actual_read_bytes=r.total,missing_charts=30-result['verified_charts'],
                  missing_QA=150-result['verified_QA'],elapsed_monotonic_seconds=time.monotonic()-start)
    body=canonical(result);require(len(body)<=128*1024,'BOUNDED_PUBLIC_SAFE_RESULT')
    with (out/'independent-result.json').open('xb') as f:f.write(body);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(passed=result['passed'],verified_charts=result['verified_charts'],verified_QA=result['verified_QA'],Dataset_GO=False)))
    return 0 if result['passed'] else 1
