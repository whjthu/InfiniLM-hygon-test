"""One frozen official HTTPS GET; receipt and partial bytes survive ordinary failures."""
import hashlib
import json
import os
import ssl
import time
import urllib.request
import urllib.error
from prepare import need,canonical

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        fp.close()
        raise urllib.error.HTTPError(req.full_url,code,'redirect forbidden; no alternate URL',headers,None)

def get_once(directory,plan,left):
    state=dict(attempted=False,headers_observed=False,http_status=None,content_length=None,content_type=None,natural_EOF=False,
        body_bytes_received=0,saved_bytes=0,partial_SHA256=hashlib.sha256(b'').hexdigest(),complete=False,close_completed=False,close_error_type=None)
    response=None;stream=None;digest=hashlib.sha256();part=directory/'source-GeoJSON-private.part'
    def save():
        raw=canonical(state);temp=directory/'http-state.tmp'
        with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        temp.replace(directory/'http-receipt-private.json')
    try:
        need(plan['URL'].startswith('https://earthquake.usgs.gov/fdsnws/event/1/query?'),'frozen official origin required')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        request=urllib.request.Request(plan['URL'],headers={'Accept':'application/json','Accept-Encoding':'identity','User-Agent':'USGS-multievent-research-v2'})
        left();state.update(attempted=True,GET_enter_epoch=time.time(),URL_SHA256=hashlib.sha256(plan['URL'].encode()).hexdigest());save()
        response=opener.open(request,timeout=min(45,left()))
        state.update(headers_observed=True,http_status=response.status,content_type=response.headers.get('Content-Type'))
        lengths=response.headers.get_all('Content-Length',[]);need(len(lengths)<=1,'duplicate Content-Length')
        if lengths:
            need(lengths[0].isdigit(),'invalid Content-Length');state['content_length']=int(lengths[0])
        save();need(response.status==200,'non200 status; no body fallback')
        need(response.geturl()==plan['URL'],'URL changed')
        need(len(response.headers.get_all('Content-Type',[]))==1 and state['content_type'].split(';')[0].strip().lower()=='application/json','JSON Content-Type required')
        need(response.headers.get('Content-Encoding','identity').lower()=='identity','encoded body forbidden')
        need(state['content_length'] is None or 0<state['content_length']<=plan['body_cap_bytes'],'Content-Length over cap/empty')
        transfer=response.headers.get_all('Transfer-Encoding',[])
        need(len(transfer)<=1 and (not transfer or transfer[0].lower()=='chunked') and not (transfer and lengths),'unsupported/ambiguous framing')
        stream=part.open('xb')
        while True:
            left();chunk=response.read(min(65536,plan['body_cap_bytes']+1-state['body_bytes_received']))
            state['body_bytes_received']+=len(chunk)
            if not chunk:state['natural_EOF']=True;break
            need(state['body_bytes_received']<=plan['body_cap_bytes'],'body exceeds4MiB+1 rejection probe')
            written=stream.write(chunk);state['saved_bytes']+=written;digest.update(chunk[:written]);state['partial_SHA256']=digest.hexdigest()
            need(written==len(chunk),'short local raw write');stream.flush();os.fsync(stream.fileno());state['fsynced_bytes']=state['saved_bytes'];save()
        need(state['saved_bytes']>0 and (state['content_length'] is None or state['saved_bytes']==state['content_length']),'EOF/Content-Length incomplete')
        left();stream.flush();os.fsync(stream.fileno());stream.close();stream=None
        state['body_complete_before_close']=True
    except BaseException as error:
        state.update(error_type=type(error).__name__,failure_phase='HTTP_OR_RAW_WRITE',private_reason=str(error)[:512])
        if isinstance(error,urllib.error.HTTPError):
            response=error;state.update(headers_observed=True,http_status=error.code,content_type=error.headers.get('Content-Type'))
            header=error.headers.get('Content-Length')
            state['content_length']=int(header) if header and header.isdigit() else None
    finally:
        if stream is not None:
            try:stream.flush();os.fsync(stream.fileno());stream.close()
            except BaseException as e:state['raw_close_error_type']=type(e).__name__
        if response is not None:
            try:response.close();state['close_completed']=True
            except BaseException as e:state['close_error_type']=type(e).__name__
        else:state['close_completed']=True
        state['complete']=bool(state.get('body_complete_before_close') and state['natural_EOF'] and state['close_completed'] and not state.get('error_type') and not state.get('close_error_type') and not state.get('raw_close_error_type'))
        state['source_outcome']='COMPLETE_RAW_SNAPSHOT' if state['complete'] else ('NO_REQUEST_ATTEMPTED' if not state['attempted'] else ('KNOWN_HTTP_OR_BODY_FAILURE' if state['headers_observed'] else 'KNOWN_LOCAL_FAILURE_REMOTE_REQUEST_EFFECT_UNKNOWN'))
        save()
    need(state['complete'],'single GET failed; keep original receipt/partial, no retry')
    left();final=directory/'source-GeoJSON-private.json';part.rename(final)
    raw=final.read_bytes();need(hashlib.sha256(raw).hexdigest()==state['partial_SHA256'] and len(raw)==state['saved_bytes'],'raw persisted SHA differs')
    return raw,state
