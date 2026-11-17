"""One preselected public train row and its official derived asset, never a model run."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import parse_qsl,unquote,urlsplit
import uuid
from job_guard import current_affinity,guarded_run

SOURCE=Path(__file__).resolve().parent;G=SOURCE.parents[2]
RUN=G/'local-only/training-CORD-viewer-derived-one-train-v1-original'


def need(ok,why):
    if not ok:raise ValueError(why)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def js(raw):
    def pairs(items):need(len(items)==len({k for k,v in items}),'Duplicate JSON key');return dict(items)
    def constant(v):raise ValueError('Nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Redirect forbidden')


def headers_gate(status,headers,kind,contract):
    need(status==200 and headers.get('Content-Encoding','identity').lower()=='identity','Only plain200 body')
    lengths=headers.get_all('Content-Length',[]);need(len(lengths)<=1,'Duplicate Content-Length');length=None
    if lengths:
        need(lengths[0].isdigit(),'Bad Content-Length');length=int(lengths[0]);need(0<length<=contract[kind+'_cap_bytes'],'Body cap rejected before read')
    if kind=='row':
        revisions=headers.get_all('X-Revision',[])
        need(revisions==[contract['revision']],'Missing/duplicate/different cache revision: prohibit image GET')
        need(headers.get_content_type()=='application/json','Row must be JSON')
    else:need(headers.get_content_type() in ('image/jpeg','image/png'),'Only official derived JPEG/PNG asset')
    return length


def asset_gate(src,row_image,ground_truth,contract):
    p=urlsplit(src);path=unquote(p.path)
    prefix='/assets/'+contract['dataset']+'/--/'+contract['revision']+'/--/'+contract['config']+'/train/'+str(contract['ordinal'])+'/image/'
    need(p.scheme=='https' and p.netloc=='datasets-server.huggingface.co' and p.username is None and not p.fragment
        and path in (prefix+'image.jpg',prefix+'image.png'),'Only exact official version/row asset; no thumbnail/crop/other host')
    queries=parse_qsl(p.query,keep_blank_values=True);keys=[k for k,v in queries]
    need(not queries or (set(keys)=={'Expires','Signature','Key-Pair-Id'} and len(keys)==3 and all(v for k,v in queries)),'Only unchanged public service-issued asset locator query')
    if queries:need(dict(queries)['Expires'].isdigit() and int(dict(queries)['Expires'])>time.time(),'Public signed locator expired')
    width,height=row_image['width'],row_image['height']
    need(type(width) is int and type(height) is int and 0<width*height<=16777216 and width>0 and height>0,'Unsafe/incomplete full-image dimensions')
    size=ground_truth['meta'].get('image_size')
    if size is not None:need(type(size.get('width')) is int and type(size.get('height')) is int and size['width']==width and size['height']==height,'Viewer asset is resized relative to released image_size; no media GET')
    return width,height


def row_gate(payload,contract):
    need(payload['partial'] is False and payload['num_rows_total']==800 and len(payload['rows'])==1,'Incomplete/partial/wrong official train inventory')
    row=payload['rows'][0];need(row['row_idx']==contract['ordinal'] and row['truncated_cells']==[],'Wrong/truncated row')
    fields={f['name']:f['type'] for f in payload['features']}
    need(len(payload['features'])==len(fields)==2 and fields['image']['_type']=='Image' and fields['ground_truth']['_type']=='Value' and fields['ground_truth']['dtype']=='string','Original two-column schema differs')
    data=row['row'];need(isinstance(data['ground_truth'],str),'Whole original ground_truth string required')
    gold=js(data['ground_truth']);meta=gold['meta']
    need(meta['version']=='2.0.0' and meta['split']=='train' and type(meta['image_id']) is int and 0<=meta['image_id']<800,'Released train metadata differs')
    need(isinstance(gold['gt_parse'],dict) and gold['gt_parse'] and not {'store_info','payment_info','etc'}.intersection(gold['gt_parse']),'Missing or removed published targets')
    width,height=asset_gate(data['image']['src'],data['image'],gold,contract)
    return gold,data['image']['src'],(width,height)


def contract_gate(c):
    order=min(range(800),key=lambda i:sha((c['seed']+':'+c['revision']+':train:'+str(i)).encode()))
    need(order==c['ordinal']==795 and c['config']=='default' and c['dataset']=='naver-clova-ix/cord-v2'
        and c['revision']=='7f0115a4b758a71d6473b8d085751692da2fef98','Pre-answer ordinal/revision/config differs')
    need(c['row_URL']=='https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix%2Fcord-v2&config=default&split=train&offset=795&length=1','Exact new row route only')


def fetch(url,path,kind,c,remaining,progress):
    # No SDK, local credentials, login, cookies, proxy, redirects, retries or alternate hosts.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    progress('GET.enter',dict(kind=kind,URL_SHA256=sha(url.encode())))
    request=urllib.request.Request(url,headers={'Accept':'application/json' if kind=='row' else 'image/jpeg,image/png','Accept-Encoding':'identity','User-Agent':'bounded-CORD-derived-train-one-v1'})
    try:response=opener.open(request,timeout=min(45,remaining()))
    except urllib.error.HTTPError as error:
        progress('GET.http_error',dict(kind=kind,status=error.code,error_code=error.headers.get('X-Error-Code'),body_read=False))
        error.close();raise
    with response as r:
        progress('GET.headers_observed',dict(kind=kind,status=r.status,Content_Type=r.headers.get('Content-Type'),
            Content_Length=r.headers.get_all('Content-Length',[]),X_Revision=r.headers.get_all('X-Revision',[]) if kind=='row' else []))
        need(r.url==url,'Response URL changed');length=headers_gate(r.status,r.headers,kind,c)
        progress('GET.headers',dict(kind=kind,status=200,declared_length=length,cache_revision=c['revision'] if kind=='row' else None))
        body=bytearray();cap=c[kind+'_cap_bytes']
        with path.open('xb') as stream:
            while True:
                remaining();need(len(body)<cap,'Body cap reached without EOF')
                chunk=r.read(min(65536,cap-len(body)))
                if not chunk:break
                body.extend(chunk);stream.write(chunk);stream.flush();progress('GET.body',dict(kind=kind,bytes=len(body)))
                if length is not None and len(body)==length:break
        need(body and (length is None or len(body)==length),'Incomplete body')
    return bytes(body)


def worker(directory,epoch,deadline):
    need(directory.resolve()==RUN.resolve(),'Only original exclusive namespace')
    started=time.monotonic();mono=started+max(0,deadline-time.time())
    def remaining():
        left=min(deadline-time.time(),mono-time.monotonic())-8;need(left>0,'Original120s epoch exhausted');return left
    while not (directory/'job-gate.json').exists():remaining();time.sleep(.02)
    gate=js((directory/'job-gate.json').read_bytes())
    need(deadline-epoch==120 and gate['deadline']==deadline and gate['CPU_count']==2 and gate['memory_limit_bytes']==2147483648
        and gate['job_affinity_verified'] is True and current_affinity()['process_mask']==gate['worker_affinity_mask'],'Actual Job/epoch differs')
    (directory/'worker-enter.lock').open('x').close()
    c=js((SOURCE/'CONTRACT.json').read_bytes());result=dict(passed=False,epoch=epoch,deadline=deadline,planned_rows=1,planned_images=1,
        actual_row_GETs=0,actual_image_GETs=0,actual_model_calls=0,Dataset_GO=False,training_GO=False,GPU_GO=False)
    def progress(phase,details):
        if phase=='GET.enter':result['actual_'+('row' if details['kind']=='row' else 'image')+'_GETs']+=1
        with (directory/'progress.jsonl').open('ab') as stream:stream.write(canonical(dict(phase=phase,time=time.time(),**details)))
    try:
        contract_gate(c)
        need(importlib.metadata.version('Pillow')=='13.0.0','Existing exact Pillow required; no install fallback')
        raw=fetch(c['row_URL'],directory/'row-response-private.json','row',c,remaining,progress);payload=js(raw)
        gold,src,dimensions=row_gate(payload,c);remaining()
        body=fetch(src,directory/'viewer-derived-image-private.body','image',c,remaining,progress)
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(body)) as image:
            need(image.format in ('JPEG','PNG') and image.size==dimensions and image.mode in ('RGB','RGBA','L','LA','P'),'Decoded full dimensions/mode differ')
            need(getattr(image,'n_frames',1)==1,'Single full receipt image required')
            expected_format='JPEG' if urlsplit(src).path.endswith('.jpg') else 'PNG'
            need(image.format==expected_format,'Asset path/decoded encoding mismatch')
            image.load();pixels=image.convert('RGBA').tobytes();remaining()
            identity=dict(body_bytes=len(body),body_SHA256=sha(body),width=image.width,height=image.height,mode=image.mode,
                canonical_decoded_pixel_mode='RGBA8',decoded_pixel_bytes=len(pixels),decoded_pixel_bytes_SHA256=sha(pixels),format=image.format,
                original_dimensions_metadata_present=gold['meta'].get('image_size') is not None,source_locator_SHA256=sha(src.encode()),
                original_Parquet_body_equivalent=False,transcode_damage_quantified=False,original_pixel_comparison_available=False)
        (directory/'gt-parse-private.json').write_bytes(canonical(gold['gt_parse']))
        (directory/'actual-image-identity-private.json').write_bytes(canonical(identity))
        # Retain full ground_truth only in the private response; do not generate a training JSONL here.
        credits=dict(dataset=c['dataset'],dataset_revision=c['revision'],cache_revision=c['revision'],row_ordinal=795,image_id=gold['meta']['image_id'],
            license='CC BY 4.0',license_URL='https://creativecommons.org/licenses/by/4.0/',publisher='NAVER CLOVA and CORD authors',
            authors=['Seunghyun Park','Seung Shin','Bado Lee','Junyeop Lee','Jaeheung Surh','Minjoon Seo','Hwalsuk Lee'],
            title='CORD: A Consolidated Receipt Dataset for Post-OCR Parsing',endorsement=False,
            source_URL='https://github.com/clovaai/cord',changes='Official Hugging Face Viewer JPEG/PNG transcode; no local resize/crop/reencode; complete frozen response/image copy',
            future_platform_rule='A100/Hygon and base/domain must use same exact frozen image body/pixels and complete gt_parse under separate domain recipe',
            official_CORD100_reproduction=False,transcode_damage_quantified=False,readability_qualified=False)
        (directory/'CREDITS-private.json').write_bytes(canonical(credits))
        result.update(passed=True,state='ONE_TRAIN_ROW_OFFICIAL_VIEWER_DERIVED_BODY_AND_PIXELS_FROZEN_ONLY',row_response_SHA256=sha(raw),
            actual_row_ordinal=795,actual_image_id=gold['meta']['image_id'],image=identity,
            gt_parse_SHA256=sha(canonical(gold['gt_parse'])),cache_revision=c['revision'],actual_metadata_API_retry=0,
            sixtyfour_dataset_qualified=False,training_qualified=False)
    except Exception as error:
        result.update(state='FAILED_ONE_SLOT_NO_FALLBACK_NO_DATASET_QUALIFICATION',error_type=type(error).__name__,error=str(error))
        progress('failure',dict(error_type=type(error).__name__,error=str(error)))
    result['wall_seconds']=time.monotonic()-started;(directory/'worker-result-private.json').write_bytes(canonical(result))
    return 0 if result['passed'] else 1


def controller(publication,claim_sha,root_go):
    need(root_go,'Root GO defaultfalse; source preparation only')
    need(re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',claim_sha),'Bad publication/claim')
    epoch=time.time();deadline=epoch+120;mono=time.monotonic()+120
    def git(*a):return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),*a],cwd=G,timeout=min(10,max(.01,deadline-time.time()-8)))
    need(not git('status','--porcelain','--untracked-files=all','--',SOURCE.relative_to(G).as_posix()).strip() and not git('diff','--cached','--name-only').strip(),'Own dirty/index')
    need(sha((SOURCE/'CLAIM.json').read_bytes())==claim_sha,'Claim changed')
    for name,pin in js((SOURCE/'MANIFEST.json').read_bytes())['files'].items():
        path=SOURCE/name;need(path.is_file() and not path.is_symlink(),'Invalid source');raw=path.read_bytes()
        need(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'] and git('show',publication+':'+path.relative_to(G).as_posix())==raw,'Published source differs')
    need(git('show',publication+':'+(SOURCE/'MANIFEST.json').relative_to(G).as_posix())==(SOURCE/'MANIFEST.json').read_bytes(),'Published MF differs')
    need(time.time()<deadline-30,'Too late for original once Job');RUN.mkdir()
    intent=dict(request=uuid.uuid4().hex,epoch=epoch,deadline=deadline,publication=publication,claim_SHA256=claim_sha,planned_rows=1,planned_images=1,maximum_GETs=2,
        original_metadata_failure_retained='5a9ee32c108548d89eecb77f324050c5',source_manifest_SHA256=sha((SOURCE/'MANIFEST.json').read_bytes()))
    (RUN/'intent.json').write_bytes(canonical(intent))
    job=guarded_run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker','--run',str(RUN),'--epoch',str(epoch),'--deadline',str(deadline)],RUN/'job-gate.json',RUN,deadline,mono)
    (RUN/'outcome.json').write_bytes(canonical(dict(**intent,known_terminal=True,job=job,Dataset_GO=False,training_GO=False,GPU_GO=False)))
    print(json.dumps(dict(request=intent['request'],returncode=job['returncode'],known_terminal=True,timed_out=job['timed_out'])))
    return job['returncode']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--run',type=Path);p.add_argument('--epoch',type=float);p.add_argument('--deadline',type=float)
    p.add_argument('--publication');p.add_argument('--claim-sha256');p.add_argument('--root-go',action='store_true');a=p.parse_args()
    raise SystemExit(worker(a.run,a.epoch,a.deadline) if a.worker else controller(a.publication,a.claim_sha256,a.root_go))
