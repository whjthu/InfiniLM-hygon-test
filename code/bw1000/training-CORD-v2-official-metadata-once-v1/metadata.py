"""Public immutable repository-file metadata only; no dataset rows/media or SDK auth."""
import hashlib
import json
from pathlib import PurePosixPath
import re
import ssl
import urllib.request
from urllib.parse import parse_qsl,unquote,urlsplit


def need(ok,why):
    if not ok:raise ValueError(why)


def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()


def strict_json(raw):
    def pairs(items):need(len(items)==len({k for k,v in items}),'Duplicate JSON key');return dict(items)
    def constant(value):raise ValueError('Nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)


def url_identity(url):
    p=urlsplit(url)
    return p.scheme.lower(),p.netloc.lower(),unquote(p.path).rstrip('/'),tuple(sorted(parse_qsl(p.query,keep_blank_values=True)))


def url_gate(contract,ledger):
    url=contract['URL'];p=urlsplit(url)
    need(p.scheme=='https' and p.netloc=='huggingface.co' and p.fragment=='' and p.username is None,'Official normal HTTPS metadata only')
    expected='/api/datasets/'+contract['dataset']+'/revision/'+contract['revision']
    need(p.path==expected and p.query=='blobs=true' and contract['GET_count']==1,'Only exact immutable metadata URL')
    need(p.path not in ledger['blocked_failed_paths_regardless_query_or_fragment'],'Failed source URL blocked')
    prior={url_identity(r['URL']) for r in ledger['web_tool_research_only']}
    need(url_identity(url) not in prior,'Exact target already requested by web; no replay')
    need(ledger['local_direct_HTTP_GET_attempts']==0 and ledger['metadata_package_URL_previously_requested_by_web_or_direct_HTTP'] is False
        and ledger['new_fixed_revision_tree_or_API_metadata_URL_failed_previously'] is False,'Known prior attempt forbids new once invocation')
    return url


def headers_gate(status,headers,cap):
    need(status==200,'Only200 metadata response; no redirect/fallback')
    need(headers.get_content_type()=='application/json' and headers.get('Content-Encoding','identity').lower()=='identity','Only plain JSON metadata')
    lengths=headers.get_all('Content-Length',[]);need(len(lengths)<=1,'Duplicate Content-Length')
    length=None
    if lengths:
        need(lengths[0].isdigit(),'Bad Content-Length');length=int(lengths[0]);need(0<length<=cap,'Metadata cap refused before body')
    need(headers.get('Link') is None,'Paginated inventory requires another scope, not automatic follow')
    return length


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('Redirect prohibited')


def get_once(contract,ledger,path,remaining,progress):
    url=url_gate(contract,ledger);cap=contract['response_body_cap_bytes']
    # Explicit no credentials, SDK, token cache discovery, proxy, redirect or retries.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    request=urllib.request.Request(url,headers={'Accept':'application/json','Accept-Encoding':'identity','User-Agent':'bounded-CORD-repository-metadata-v1'})
    progress('GET.enter',dict(URL=url,attempt=1,body_cap=cap))
    with opener.open(request,timeout=min(45,remaining())) as response:
        need(response.url==url,'Response URL changed');length=headers_gate(response.status,response.headers,cap)
        progress('GET.headers',dict(status=200,Content_Length=length,normal_TLS=True))
        body=bytearray()
        with path.open('xb') as stream:
            while True:
                remaining();need(len(body)<cap,'Reached metadata cap without EOF')
                chunk=response.read(min(65536,cap-len(body)))
                if not chunk:break
                body.extend(chunk);stream.write(chunk);stream.flush();progress('GET.body',dict(actual_bytes=len(body)))
                if length is not None and len(body)==length:break
        need(len(body)>0 and (length is None or len(body)==length),'Incomplete metadata body')
    return bytes(body),dict(URL=url,status=200,bytes=len(body),SHA256=hashlib.sha256(body).hexdigest(),Content_Length=length)


def inventory(payload,contract):
    need(isinstance(payload,dict) and payload['id']==contract['dataset'] and payload['sha']==contract['revision'],'Wrong repository or mutable revision')
    need(payload.get('private') is False and payload.get('gated') in (False,None),'No private/authenticated data scope')
    need(payload.get('cardData',{}).get('license')=='cc-by-4.0','Published dataset license metadata missing/changed')
    siblings=payload['siblings'];need(isinstance(siblings,list) and 6<=len(siblings)<=100,'Unbounded/absent sibling metadata')
    seen=set();files=[]
    for entry in siblings:
        name=entry['rfilename'];p=PurePosixPath(name)
        need(isinstance(name,str) and not p.is_absolute() and '..' not in p.parts and name not in seen,'Unsafe/duplicate file path');seen.add(name)
        if not name.endswith('.parquet'):continue
        size=entry['size'];blob=entry['blobId'];lfs=entry.get('lfs')
        need(name in contract['expected_parquet_paths'] and type(size) is int and 12<size<=1073741824,'Unexpected parquet path/size')
        need(isinstance(blob,str) and re.fullmatch('[0-9a-f]{40}',blob) is not None and isinstance(lfs,dict),'Missing Git blob/LFS metadata')
        need(type(lfs['size']) is int and lfs['size']==size and isinstance(lfs['sha256'],str) and re.fullmatch('[0-9a-f]{64}',lfs['sha256']) is not None
            and type(lfs['pointerSize']) is int and 0<lfs['pointerSize']<=4096,'LFS identity malformed')
        files.append(dict(path=name,size_bytes=size,git_blob_OID=blob,LFS_SHA256=lfs['sha256'],LFS_pointer_bytes=lfs['pointerSize'],
            actual_file_body_hashed=False,Range_support='UNKNOWN_NO_RANGE_SENT',footer_or_rowgroups_read=False))
    need({f['path'] for f in files}==set(contract['expected_parquet_paths']) and len(files)==6,'All six official file records required')
    need(len({f['LFS_SHA256'] for f in files})==6,'Duplicate LFS file identity; keep metadata failure')
    return dict(state='REPOSITORY_FILE_METADATA_ONLY_RANGE_AND_ROWGROUPS_UNKNOWN',dataset=contract['dataset'],revision=contract['revision'],
        parquet_files=sorted(files,key=lambda f:f['path']),parquet_file_count=6,reported_parquet_total_bytes=sum(f['size_bytes'] for f in files),
        publisher_advertised_full_download_bytes=2307284272,metadata_license='cc-by-4.0',source_blob_identity_only=True,
        actual_media_bytes=0,actual_parquet_body_bytes=0,actual_row_labels_read=0,Range_qualified=False,footer_offsets=None,rowgroup_inventory=None,
        common_framework_binding='future native qwen3_vl Swift messages/images; not a loader, training loop or dataset qualification',
        Dataset_GO=False,training_GO=False,GPU_GO=False)
