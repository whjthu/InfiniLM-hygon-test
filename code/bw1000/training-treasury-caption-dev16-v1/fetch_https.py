"""One public HTTPS Range; reject response headers before reading any body."""
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

URL='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
EXPECTED_ETAG='"f4bbac642086de4f52a3fdda2de5fa2c"'
START,END,TOTAL=159287046,178103193,252907541
BYTES=18816148


def require(ok,why):
    if not ok:raise ValueError(why)


def remaining(deadline,monotonic_end):
    require(all(math.isfinite(x) for x in (deadline,monotonic_end)),'invalid original public-fetch deadline')
    value=min(deadline-time.time(),monotonic_end-time.monotonic())
    require(value>0,'original public-fetch epoch exhausted')
    return value


def validate_response(response):
    require(response.status==206,'HTTP206 required; body not read')
    require(response.geturl()==URL,'redirect/final official URL differs; body not read')
    headers=response.headers
    require(headers.get_all('Content-Range')==[f'bytes {START}-{END}/{TOTAL}'],
            'exact single Content-Range required; body not read')
    require(headers.get_all('Content-Length')==[str(BYTES)],'exact single Content-Length required; body not read')
    require(headers.get_all('ETag')==[EXPECTED_ETAG],'exact single object ETag required; body not read')
    require(headers.get_all('Content-Encoding') in (None,['identity'])
            and headers.get_all('Transfer-Encoding') is None,'compressed/chunked transport rejected before body')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise urllib.error.HTTPError(req.full_url,code,'redirect prohibited',headers,None)


def fetch(directory,deadline,monotonic_end):
    """No SSH/config/model/private input reads; caller already owns exclusive directory."""
    parts=urlsplit(URL)
    require(parts.scheme=='https' and parts.hostname=='s3.us-east-1.amazonaws.com' and parts.port in (None,443)
            and parts.username is None and parts.password is None and not parts.query and not parts.fragment,
            'official HTTPS URL identity changed')
    state=dict(URL=URL,range_start=START,range_end=END,expected_body_bytes=BYTES,HTTP_open_attempts=0,
        body_bytes_read=0,body_bytes_saved=0,deadline=deadline,expected_ETag=EXPECTED_ETAG,official_TLS_default_validation=True,
        proxy_environment_used=False,redirects_allowed=False,retries_allowed=False,complete=False)
    def save():
        (directory/'fetch-progress.json').write_text(json.dumps(state,sort_keys=True),encoding='utf8')
    save()
    response=None
    try:
        context=ssl.create_default_context()
        require(context.check_hostname and context.verify_mode==ssl.CERT_REQUIRED,'normal TLS validation disabled')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),
                                          urllib.request.HTTPSHandler(context=context))
        request=urllib.request.Request(URL,headers={'Range':f'bytes={START}-{END}',
            'Accept-Encoding':'identity','User-Agent':'InfiniLM-public-caption-dev16-v1'},method='GET')
        state['HTTP_open_attempts']=1;save()
        response=opener.open(request,timeout=min(15,remaining(deadline,monotonic_end)))
        validate_response(response)
        state['response_headers_qualified']=True;save()
        partial=directory/'captions_train2017.local-record.partial'
        final=directory/'captions_train2017.local-record.bin'
        require(not partial.exists() and not final.exists(),'member output already exists')
        digest=hashlib.sha256()
        with partial.open('xb') as stream:
            while state['body_bytes_read']<BYTES:
                remaining(deadline,monotonic_end)
                chunk=response.read(min(256*1024,BYTES-state['body_bytes_read']))
                require(chunk,'truncated exact public Range')
                require(len(chunk)<=BYTES-state['body_bytes_read'],'response overrun')
                state['body_bytes_read']+=len(chunk);stream.write(chunk);digest.update(chunk)
                state['body_bytes_saved']+=len(chunk);save()
            stream.flush()
        require(state['body_bytes_saved']==BYTES,'member byte denominator differs')
        partial.rename(final)
        state.update(complete=True,member_SHA256=digest.hexdigest(),member_file=final.name)
        save()
        return state
    except BaseException as error:
        if isinstance(error,urllib.error.HTTPError):error.close()
        state['failure_type']=type(error).__name__;save();raise
    finally:
        if response is not None:response.close()
