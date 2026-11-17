"""Exactly one missing suffix GET; no API capable of requesting the acquired prefix."""
import hashlib
import json
import math
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from chain import require,ETAG,TAIL_START,TAIL_END,TAIL_BYTES,PREFIX_BYTES,FULL_BYTES

URL='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
TOTAL=252907541


def left(deadline,monotonic_end):
    require(all(math.isfinite(v) for v in (deadline,monotonic_end)),'invalid original tail clock')
    value=min(deadline-time.time(),monotonic_end-time.monotonic())-7
    require(value>0,'new tail epoch exhausted');return value


def validate_headers(response):
    require(response.status==206 and response.geturl()==URL,'suffix must be206 at fixedURL, beforebody')
    h=response.headers
    require(h.get_all('Content-Range')==[f'bytes {TAIL_START}-{TAIL_END}/{TOTAL}'],
            'suffix exact single ContentRange differs, beforebody')
    require(h.get_all('Content-Length')==[str(TAIL_BYTES)],'suffix exact single ContentLength differs, beforebody')
    require(h.get_all('ETag')==[ETAG],'suffix same single ETag required, beforebody')
    require(h.get_all('Content-Encoding') in (None,['identity']) and h.get_all('Transfer-Encoding') is None,
            'suffix transport encoding rejected, beforebody')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise urllib.error.HTTPError(req.full_url,code,'tail redirects prohibited',headers,None)


def acquire_tail(directory,assembled,full_digest,deadline,monotonic_end):
    parts=urlsplit(URL)
    require(parts.scheme=='https' and parts.hostname=='s3.us-east-1.amazonaws.com' and parts.port in (None,443)
            and parts.username is None and parts.password is None and not parts.query and not parts.fragment,'fixedTLSURL differs')
    require(assembled.is_file() and not assembled.is_symlink() and assembled.stat().st_size==PREFIX_BYTES,'new copied prefix differs')
    before=assembled.lstat();tail_digest=hashlib.sha256()
    progress=dict(URL=URL,range_start=TAIL_START,range_end=TAIL_END,expected_suffix_bytes=TAIL_BYTES,
        expected_ETag=ETAG,preserved_original_prefix_bytes=PREFIX_BYTES,HTTP_prefix_bytes_requested=0,
        HTTP_open_attempts=0,suffix_bytes_read=0,suffix_bytes_saved=0,assembled_bytes=PREFIX_BYTES,
        deadline=deadline,complete=False,normal_TLS=True,retries=False,redirects=False,proxy_environment=False)
    def save():(directory/'tail-progress.json').write_text(json.dumps(progress,sort_keys=True),encoding='utf8')
    save();response=None
    try:
        context=ssl.create_default_context()
        require(context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED,'normalTLS validation disabled')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=context))
        request=urllib.request.Request(URL,headers={'Range':f'bytes={TAIL_START}-{TAIL_END}','If-Match':ETAG,
            'Accept-Encoding':'identity','User-Agent':'InfiniLM-caption-dev16-missing-tail-v1'},method='GET')
        progress['HTTP_open_attempts']=1;save()
        response=opener.open(request,timeout=min(15,left(deadline,monotonic_end)))
        validate_headers(response);progress['response_headers_qualified']=True;save()
        with assembled.open('ab') as stream:
            while progress['suffix_bytes_read']<TAIL_BYTES:
                left(deadline,monotonic_end)
                chunk=response.read(min(256*1024,TAIL_BYTES-progress['suffix_bytes_read']))
                require(chunk and len(chunk)<=TAIL_BYTES-progress['suffix_bytes_read'],'truncated/overrun suffix')
                progress['suffix_bytes_read']+=len(chunk);tail_digest.update(chunk);full_digest.update(chunk)
                stream.write(chunk);progress['suffix_bytes_saved']+=len(chunk)
                progress['assembled_bytes']=PREFIX_BYTES+progress['suffix_bytes_saved'];save()
            stream.flush()
        after=assembled.lstat()
        require(after.st_dev==before.st_dev and after.st_ino==before.st_ino and after.st_size==FULL_BYTES
                and progress['suffix_bytes_saved']==TAIL_BYTES,'new assembled byte/inode identity differs')
        final=directory/'captions_train2017.local-record.bin';require(not final.exists(),'new assembled target exists')
        assembled.rename(final)
        progress.update(complete=True,suffix_SHA256=tail_digest.hexdigest(),assembled_SHA256=full_digest.hexdigest(),
                        member_file=final.name,actual_new_HTTP_body_bytes=TAIL_BYTES)
        save();return progress
    except BaseException as error:
        if isinstance(error,urllib.error.HTTPError):error.close()
        progress['failure_type']=type(error).__name__;save();raise
    finally:
        if response is not None:response.close()
