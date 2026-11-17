"""Fixed public keyless Flickr oEmbed; no HTML/image fetch, private API or JSON-LD guess."""
import hashlib
import json
import math
import re
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlsplit

ALPHABET = '123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
MAX_BYTES = 65536
ENDPOINT = 'https://www.flickr.com/services/oembed/'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def short_url(photo_id):
    require(isinstance(photo_id,str) and re.fullmatch('[1-9][0-9]*',photo_id), 'Invalid frozen numeric Flickr photoID')
    value = int(photo_id); result = ''
    while value:
        value, digit = divmod(value,58); result = ALPHABET[digit]+result
    return 'https://flic.kr/p/'+result


def endpoint(photo_id):
    return ENDPOINT+'?'+urlencode({'format':'json','url':short_url(photo_id)})


def left(deadline, monotonic_end):
    value=min(deadline-time.time(),monotonic_end-time.monotonic())-7
    require(value>0,'Original600s credits epoch exhausted'); return value


def photo_owner(url, photo_id=None):
    part=urlsplit(url or '')
    require(part.scheme in ('http','https') and part.hostname in ('www.flickr.com','flickr.com')
        and part.port in (None,80,443) and part.username is None and part.password is None
        and not part.query and not part.fragment,'Non-primary Flickr identity URL')
    pattern=r'/photos/([^/]+)/'+re.escape(photo_id)+r'/?' if photo_id else r'/(?:photos|people)/([^/]+)/?'
    match=re.fullmatch(pattern,part.path); require(match,'Canonical Flickr photo/author path differs')
    require(re.fullmatch(r'[A-Za-z0-9_@.-]+',match[1]),'Unsafe Flickr owner identity')
    return match[1]


def license_url_qualified(payload):
    accepted=('http://creativecommons.org/licenses/by/2.0/','https://creativecommons.org/licenses/by/2.0/')
    if payload.get('license_url') not in accepted:
        return False
    # Unspecified extensions cannot silently override/conflict with the explicit URL.
    for key in ('license','licence','licence_url','license_name','license_type','rights','copyright'):
        if key in payload and payload[key] not in accepted:
            return False
    return True


def extract(payload,photo_id):
    require(isinstance(payload,dict) and payload.get('version')=='1.0' and payload.get('type')=='photo'
        and payload.get('provider_name')=='Flickr','Optout/link/nonphoto/nonFlickr oEmbed response')
    name,title,author=payload.get('author_name'),payload.get('title'),payload.get('author_url')
    require(isinstance(name,str) and 0<len(name.strip())<=512 and isinstance(title,str) and 0<len(title.strip())<=2048,
        'Actual author/title missing; no invented credit')
    owner=photo_owner(author)
    image=urlsplit(payload.get('url',''))
    match=re.fullmatch(r'/(?:[0-9]+/){1,3}([0-9]+)_[0-9a-f]+(?:_[a-z0-9]+)?\.(?:jpg|jpeg|png|gif)',image.path,re.I)
    require(image.scheme in ('http','https') and re.fullmatch(r'(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com|farm[0-9]+\.static\.flickr\.com',image.hostname or '')
        and match and match[1]==photo_id,'Actual response image photoID differs from frozen request')
    canonical='https://www.flickr.com/photos/'+owner+'/'+photo_id+'/'
    if 'web_page' in payload:
        require(photo_owner(payload['web_page'],photo_id)==owner,'Actual canonical photo owner/photoID mismatch')
        canonical_source='actual_oEmbed_web_page'
    else:
        canonical_source='derived_from_actual_author_owner_and_bound_photoID_using_official_URL_rule'
    raw_license=payload.get('license_url')
    explicit = license_url_qualified(payload)
    # No numeric app/license mapping. COCO license4 never supplies this missing field.
    return dict(author_name=name,author_url=author,title=title,canonical_photo_URL=canonical,
        canonical_URL_source=canonical_source,photoID_association_verified=True,
        author_title_qualified=True,actual_license_url=raw_license if isinstance(raw_license,str) else None,
        license_sourceproof_qualified=explicit,license_status='ACTUAL_EXPLICIT_CC_BY2_URL' if explicit else 'PENDING_OR_CONFLICTING_NO_ASSUMPTION',
        credits_sourceproof_qualified=explicit,license_standard_field_guaranteed=False,
        source='actual_public_Flickr_oEmbed_JSON',COCO_license4_used_as_current_license_proof=False,
        public_photo_page_fetched=False,response_image_fetched=False)


def unique_object(pairs):
    require(len(pairs)==len({k for k,v in pairs}),'Duplicate JSON evidence key rejected')
    return dict(pairs)


def invalid_constant(value):
    raise ValueError('Nonstandard numeric JSON constant rejected')


def finite_float(value):
    number=float(value)
    require(math.isfinite(number),'JSON floating number is nonfinite after parsing')
    return number


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def acquire(photo_id,path,deadline,monotonic_end,progress):
    url=endpoint(photo_id); result=dict(state='NOT_STARTED',HTTP_open_attempts=0,bytes_read=0,bytes_saved=0,
        author_title_qualified=False,license_sourceproof_qualified=False,credits_sourceproof_qualified=False,
        URL=url,retry=False,no_backfill=True)
    response=None
    try:
        require(left(deadline,monotonic_end)>=15,'Insufficient original epoch to start single oEmbed GET')
        tls=ssl.create_default_context(); require(tls.check_hostname and tls.verify_mode==ssl.CERT_REQUIRED,'Normal TLS required')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=tls))
        request=urllib.request.Request(url,headers={'Accept-Encoding':'identity','Accept':'application/json','User-Agent':'InfiniLM-caption-dev16-public-credits-v1'},method='GET')
        result.update(state='OPEN',HTTP_open_attempts=1,normal_TLS=True,redirects=False,proxy_environment=False);progress(result)
        response=opener.open(request,timeout=min(15,left(deadline,monotonic_end)))
        require(response.status==200 and response.geturl()==url,'Only fixed oEmbed exact200; no redirect/fallback')
        h=response.headers; types=h.get_all('Content-Type')
        require(types is not None and len(types)==1 and types[0].split(';')[0].strip().lower() in ('application/json','application/json+oembed'), 'Only JSON content before body')
        require(h.get_all('Content-Encoding') in (None,['identity']) and h.get_all('Content-Range') is None,'Encoded/ranged JSON rejected before body')
        lengths=h.get_all('Content-Length');transfer=h.get_all('Transfer-Encoding')
        require(lengths is None or (len(lengths)==1 and re.fullmatch('[1-9][0-9]*',lengths[0]) and int(lengths[0])<=MAX_BYTES),'Missing-valid-or-single bounded JSON length only')
        require(transfer in (None,['chunked']) and not (lengths is not None and transfer is not None),'Ambiguous HTTP framing rejected before body')
        length=int(lengths[0]) if lengths else None
        result['response_headers']=dict(status=200,Content_Type=types[0],Content_Length=length,Transfer_Encoding=transfer,Content_Encoding=h.get_all('Content-Encoding'))
        result['headers_qualified_before_body']=True;progress(result)
        left(deadline,monotonic_end)
        raw=response.read(length if length is not None else MAX_BYTES)
        result['bytes_read']=len(raw)
        require(0<len(raw)<=MAX_BYTES and (len(raw)==length if length is not None else len(raw)<MAX_BYTES),
            'Truncated known-length or ambiguous max-size unknown-length JSON; no extra read')
        with path.open('xb') as stream:stream.write(raw)
        result['bytes_saved']=len(raw);result['body_SHA256']=hashlib.sha256(raw).hexdigest();progress(result)
        evidence=extract(json.loads(raw.decode('utf8'),object_pairs_hook=unique_object,parse_constant=invalid_constant,parse_float=finite_float),photo_id)
        result.update(state='ACTUAL_AUTHOR_TITLE_RECORDED_LICENSE_SEPARATELY_ASSESSED',evidence=evidence,
            author_title_qualified=True,license_sourceproof_qualified=evidence['license_sourceproof_qualified'],
            credits_sourceproof_qualified=evidence['credits_sourceproof_qualified'])
    except BaseException as error:
        result.update(state='STOP_THIS_FIXED_SLOT_NO_RETRY',error_type=type(error).__name__)
        if isinstance(error,urllib.error.HTTPError):
            result['actual_HTTP_error_status']=error.code;error.close()
    finally:
        if response is not None:response.close()
        progress(result)
    return result
