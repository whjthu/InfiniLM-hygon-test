"""Exactly one normal-TLS GET per frozen endpoint; no redirects/retries/fallback."""
import hashlib
import ssl
import urllib.request
from urllib.parse import urlsplit

from contracts import need


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Redirect forbidden')


def headers_guard(status,headers,cap):
    need(status==200,'HTTP status must be200')
    lengths=headers.get_all('Content-Length',[])
    need(len(lengths)<=1,'Duplicate Content-Length')
    length=None
    if lengths:
        need(lengths[0].isdigit(),'Invalid Content-Length');length=int(lengths[0])
        need(0<length<=cap,'HTTP body exceeds endpoint cap before read')
    need(headers.get('Content-Encoding','identity').lower()=='identity','Encoded body forbidden')
    need(headers.get_content_type() in ('application/json','text/json'),'Not JSON Content-Type')
    return length


def fetch_one(endpoint,directory,check,progress):
    url=endpoint['URL'];parts=urlsplit(url);cap=endpoint['cap_bytes']
    need(parts.scheme=='https' and parts.netloc=='api.worldbank.org' and parts.username is None and parts.fragment=='','Frozen HTTPS official API only')
    remaining=check()
    opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request=urllib.request.Request(url,headers={'Accept':'application/json','Accept-Encoding':'identity','User-Agent':'research-bounded-worldbank-macro-v1'})
    with opener.open(request,timeout=min(20,remaining)) as response:
        need(response.url==url,'Response URL changed');length=headers_guard(response.status,response.headers,cap)
        progress('headers',dict(URL=url,status=response.status,length=length,cap=cap))
        body=bytearray()
        with directory.open('xb') as stream:
            while True:
                check()
                # Unknown-length bodies cannot be probed beyond cap. Reject an exact-full cap too.
                need(len(body)<cap,'Unknown/overlong body reached cap without EOF')
                chunk=response.read(min(65536,cap-len(body)))
                if not chunk:break
                stream.write(chunk);stream.flush();body.extend(chunk)
                progress('body',dict(saved_bytes=len(body),cap=cap))
                if length is not None and len(body)==length:break
        need(len(body)>0 and (length is None or len(body)==length),'Incomplete declared JSON')
        return bytes(body),dict(URL=url,status=200,bytes=len(body),SHA256=hashlib.sha256(body).hexdigest(),declared_length=length)
