"""Independent fixed16 public metadata terminal audit; no producer engine import or HTTP."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from urllib.parse import urlencode,urlsplit

G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
RUN=G/'local-only/training-treasury-caption-dev16-oembed-credits-v1-original'
SOURCE=G/'code/bw1000/training-treasury-caption-dev16-oembed-credits-v1'
ROSTER_SHA='96eaeb4b4bd5f52bcf6fbd39370f61fc1ed24ea22842dc77e15c69a5c1885627'


def need(ok,why):
    if not ok:raise ValueError(why)


def unique(pairs):
    need(len(pairs)==len({k for k,v in pairs}),'Duplicate JSON evidence keys');return dict(pairs)


def invalid_constant(value):
    raise ValueError('Nonstandard JSON numeric constant')


def parse_finite_decimal(text):
    parsed=float(text)
    need(math.isfinite(parsed),'Overflow/nonfinite JSON decimal')
    return parsed


def small(path,cap=131072,sha=None,parse=True):
    before=path.lstat();need(stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size<=cap,'Unsafe metadata evidence')
    raw=path.read_bytes();after=path.lstat()
    need(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Metadata changed')
    pin=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest());need(sha is None or pin['SHA256']==sha,'Metadata SHA differs')
    return (json.loads(raw,object_pairs_hook=unique,parse_constant=invalid_constant,parse_float=parse_finite_decimal) if parse else raw),pin


def expected_url(identifier):
    need(re.fullmatch('[1-9][0-9]*',identifier),'Invalid frozen photoID');n=int(identifier);digits=[]
    alphabet='123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
    while n:
        n,d=divmod(n,58);digits.append(alphabet[d])
    short='https://flic.kr/p/'+''.join(reversed(digits))
    return 'https://www.flickr.com/services/oembed/?'+urlencode(dict(format='json',url=short))


def owner(url,photo=None):
    part=urlsplit(url or '')
    need(part.scheme in ('http','https') and part.hostname in ('www.flickr.com','flickr.com')
        and part.port in (None,80,443) and part.username is None and part.password is None
        and not part.query and not part.fragment,'Nonprimary attribution URL')
    pattern=r'/photos/([A-Za-z0-9_@.-]+)/'+re.escape(photo)+r'/?' if photo else r'/(?:photos|people)/([A-Za-z0-9_@.-]+)/?'
    match=re.fullmatch(pattern,part.path);need(match,'Photo/author URL mismatch');return match[1]


def independent_license_proof(body):
    urls=('https://creativecommons.org/licenses/by/2.0/','http://creativecommons.org/licenses/by/2.0/')
    if body.get('license_url') not in urls:return False
    declarations=('copyright','rights','license_type','license_name','licence_url','licence','license')
    return all(body[key] in urls for key in declarations if key in body)


def persisted_framing(record,actual_bytes):
    """Check saved pre-body fields against actual body; never claim a live HTTP observation."""
    need(type(record.get('HTTP_open_attempts')) is int and record['HTTP_open_attempts']==1,'Expected one saved open attempt')
    header=record.get('response_headers')
    need(type(header) is dict and header.get('status')==200,'Saved HTTP status differs')
    content=header.get('Content_Type')
    need(type(content) is str and content.split(';')[0].strip().lower() in ('application/json','application/json+oembed'),'Saved MIME differs')
    length=header.get('Content_Length');transfer=header.get('Transfer_Encoding');encoding=header.get('Content_Encoding')
    need(length is None or (type(length) is int and 0<length<=65536),'Saved length differs')
    need(transfer in (None,['chunked']) and not (length is not None and transfer is not None)
         and encoding in (None,['identity']),'Saved framing/encoding conflict')
    need(type(actual_bytes) is int and 0<actual_bytes<=65536
         and (actual_bytes==length if length is not None else actual_bytes<65536),'Actual bytes differ from saved framing')
    return True


def credit(body,identifier):
    need(isinstance(body,dict) and body.get('type')=='photo' and body.get('version')=='1.0'
        and body.get('provider_name')=='Flickr','Optout/link/nonphoto evidence')
    name,title=body.get('author_name'),body.get('title')
    need(isinstance(name,str) and 0<len(name.strip())<=512 and isinstance(title,str) and 0<len(title.strip())<=2048,'Missing actual author/title')
    subject=owner(body.get('author_url'));photo=urlsplit(body.get('url',''))
    match=re.fullmatch(r'/(?:[0-9]+/){1,3}([0-9]+)_[0-9a-f]+(?:_[a-z0-9]+)?\.(?:jpg|jpeg|png|gif)',photo.path,re.I)
    need(photo.scheme in ('http','https') and re.fullmatch(r'(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com|farm[0-9]+\.static\.flickr\.com',photo.hostname or '')
        and match and match[1]==identifier,'Response does not bind actual frozen photoID')
    if 'web_page' in body:need(owner(body['web_page'],identifier)==subject,'Photo/author canonical association differs')
    license_url=body.get('license_url')
    explicit=independent_license_proof(body)
    identity=dict(author_name=name,author_url=body['author_url'],title=title,
        canonical='https://www.flickr.com/photos/'+subject+'/'+identifier+'/',license_url=license_url)
    return dict(author_title_qualified=True,explicit_same_CC_BY2_license_sourceproof=explicit,
        credits_sourceproof_qualified=explicit,credit_identity_SHA256=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest())


def main(a):
    need(re.fullmatch('[0-9a-f]{40}',a.publication) and re.fullmatch('[0-9a-f]{32}',a.request)
        and all(re.fullmatch('[0-9a-f]{64}',v) for v in (a.claim_sha256,a.source_mf_sha256)),'Invalid actual invocation identity')
    _,success_pin=small(G/'results/bw1000/2026-11-17/training-caption-dev16-complete-member-and-private-roster-independent-terminal.json',sha='47bfd69ac07f08ffae859f0abff58f3abcf2bf74fb5a00d0a42fc6e131b2230a')
    media,media_pin=small(G/'local-only/training-treasury-caption-dev16-v1/media-original-independent-result.json',sha='9becb5ef8a50f5545af1f4bd387e06bf46d5b6415bf7401e0b3aa570f195c94f')
    need(media['passed'] is True and media['independently_qualified_media_images']==16,'Original media baseline differs')
    roster,roster_pin=small(G/'local-only/training-treasury-caption-dev16-final-tail-v1-original/selection/roster.json',32768,ROSTER_SHA)
    need(len(roster)==len({r['flickr_photo_id'] for r in roster})==16,'Frozen16 denominator differs')
    mf,mf_pin=small(SOURCE/'MF.json',sha=a.source_mf_sha256);claim,claim_pin=small(SOURCE/'CLAIM.json',sha=a.claim_sha256)
    need({p.name for p in SOURCE.iterdir() if p.is_file()}==set(mf)|{'MF.json'},'Actual source inventory differs')
    for name,pin in mf.items():
        need(Path(name).name==name,'Unsafe source path');_,actual=small(SOURCE/name,1048576,parse=False);need(actual==pin,'Actual source pin differs')
    intent,ip=small(RUN/'intent.json');outcome,op=small(RUN/'outcome.json')
    need(intent['request']==outcome['request']==a.request and intent['publication']==a.publication and intent['claim_SHA256']==a.claim_sha256
        and intent['epoch']==outcome['epoch']==a.epoch and intent['deadline']==outcome['deadline']==a.epoch+600
        and intent['maximum_HTTP_open_attempts']==16 and intent['maximum_concurrent_requests']==4
        and intent['maximum_total_body_bytes']==1048576 and intent['maximum_bytes_per_image']==65536
        and intent['replay_allowed'] is False,'Original credits context/budget differs')
    guard=outcome.get('guarded_process');worker=outcome.get('worker');worker_pin=None;progress_pin=None;records={}
    if guard is not None:
        need(guard['deadline']==a.epoch+600 and guard['WindowsJobObject_limits_verified'] is True
            and guard['job_CPU_affinity_verified'] is True and guard['worker_CPU_affinity_verified_before_gate'] is True
            and guard['worker_CPU_count']==2 and int(guard['worker_affinity_mask']).bit_count()==2
            and guard['process_commit_limit_bytes']==guard['job_commit_limit_bytes']==2147483648
            and type(guard['returncode']) is int and type(guard['timed_out']) is bool and guard['wall_seconds']<=600,
            'Actual terminal Job2CPU2GiB600 differs')
    else:need(outcome['passed'] is False and outcome.get('error_type'),'No terminal/prelaunch failure evidence')
    if worker is not None:
        actual,worker_pin=small(RUN/'worker-result.json');need(actual==worker,'Actual embedded worker differs')
        need(guard is not None and worker['actual_worker_CPU_count']==2 and worker['actual_worker_affinity_mask']==guard['worker_affinity_mask']
            and worker['planned_images']==16 and worker['maximum_concurrent_requests']==4 and worker['maximum_body_bytes_per_slot']==65536,
            'Worker limits differ')
        records={i:r for i,r in enumerate(worker['image_records'])} if 'image_records' in worker else {int(i):r for i,r in worker.get('surviving_attempt_records',{}).items()}
    if (RUN/'progress.json').exists():
        progress,progress_pin=small(RUN/'progress.json');need(progress['deadline']==a.epoch+600 and progress['planned_images']==16,'Actual progress differs')
        if not records:records={int(i):r for i,r in progress['records'].items()}
    need(all(0<=i<16 for i in records),'Unexpected credits slot')
    slots=[];attempts=body_bytes=0;actual_names=set()
    for i,row in enumerate(roster):
        path=RUN/'raw-json'/f'{i:02d}.json';actual_names.add(path.name);r=records.get(i,{})
        need(r.get('HTTP_open_attempts',0) in (0,1) and 0<=r.get('bytes_saved',0)<=r.get('bytes_read',0)<=65536,'Slot attempts/bytes exceed claim')
        attempts+=r.get('HTTP_open_attempts',0);body_bytes+=r.get('bytes_read',0)
        value=dict(slot=i,author_title_qualified=False,explicit_same_CC_BY2_license_sourceproof=False,credits_sourceproof_qualified=False)
        if path.exists():
            try:
                raw,pin=small(path,65536,parse=False);value['actual_body_pin']=pin
                need(pin['SHA256']==r['body_SHA256'] and pin['bytes']==r['bytes_saved']==r['bytes_read']
                    and r['URL']==expected_url(row['flickr_photo_id']) and r['normal_TLS'] is True
                    and r['redirects'] is False and r['proxy_environment'] is False and r['headers_qualified_before_body'] is True
                    and r['response_headers']['status']==200,'Actual oneGET/source body identity differs')
                persisted_framing(r,pin['bytes'])
                body=json.loads(raw.decode('utf8'),object_pairs_hook=unique,parse_constant=invalid_constant,parse_float=parse_finite_decimal);value.update(credit(body,row['flickr_photo_id']))
            except (ValueError,KeyError,TypeError) as error:value['failure_type']=type(error).__name__
        slots.append(value)
    if (RUN/'raw-json').exists():need(all(p.is_file() and not p.is_symlink() and p.name in actual_names for p in (RUN/'raw-json').iterdir()),'Unexpected raw metadata file')
    author=sum(v['author_title_qualified'] for v in slots);license_count=sum(v['explicit_same_CC_BY2_license_sourceproof'] for v in slots);credits=sum(v['credits_sourceproof_qualified'] for v in slots)
    need(attempts<=16 and body_bytes<=1048576,'Whole16 credits budget exceeded')
    if worker and 'image_records' in worker:
        need(author==worker['author_title_qualified'] and 16-author==worker['missing_author_title']
            and license_count==worker['explicit_same_CC_BY2_license_sourceproof'] and credits==worker['credits_sourceproof_qualified']
            and attempts==worker['actual_HTTP_open_attempts'] and body_bytes==worker['actual_HTTP_body_bytes'],'Independent author/license/full16 denominator differs')
    if outcome['passed'] is True:need(guard is not None and guard['returncode']==0 and guard['timed_out'] is False and worker is not None
        and worker['passed'] is True and author==16 and worker['actual_worker_affinity_unchanged'] is True,'Success lacks full terminal author evidence')
    need(outcome['replay_allowed'] is False and outcome['no_failure_backfill'] is True and outcome['further_local_credits_retry_allowed'] is False
        and all(outcome[k] is False for k in ('evaluation_ready','Dataset_GO','training_GO','GPU_GO')),'Final no-retry/qualification limitations differ')
    result=dict(passed=True,producer_passed=outcome['passed'],request=a.request,source_publication=a.publication,
        planned_images=16,author_title_qualified=author,missing_author_title=16-author,
        explicit_same_CC_BY2_license_sourceproof=license_count,license_pending_or_conflicting=16-license_count,credits_sourceproof_qualified=credits,
        actual_HTTP_open_attempts=attempts,actual_HTTP_body_bytes=body_bytes,actual_guard=guard,private_ordered_slots=slots,
        pins=dict(metadata_success=success_pin,media=media_pin,roster=roster_pin,source_MF=mf_pin,claim=claim_pin,intent=ip,outcome=op,worker=worker_pin,progress=progress_pin),
        live_response_headers_requeried=False,headers_evidence='Independent validation of persisted producer fields versus actual body; not live HTTP observation',
        license_optional_extension_not_standard_guarantee=True,COCO_license4_used_as_current_license=False,
        new_HTTP_requests=0,new_images_fetched=0,old_partials_opened=False,reference_files_opened=False,
        native_processor='NOT_RUN',scorer_calls=0,training_updates=0,evaluation_ready=False,Dataset_GO=False,training_GO=False,GPU_GO=False)
    need(a.output.resolve().is_relative_to((G/'local-only').resolve()) and not a.output.exists(),'Output must be new local-only')
    with a.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('passed','producer_passed','planned_images','author_title_qualified','missing_author_title','explicit_same_CC_BY2_license_sourceproof','credits_sourceproof_qualified','new_HTTP_requests','Dataset_GO','training_GO','GPU_GO')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);p.add_argument('--claim-sha256',required=True);p.add_argument('--source-mf-sha256',required=True)
    p.add_argument('--request',required=True);p.add_argument('--epoch',type=float,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
