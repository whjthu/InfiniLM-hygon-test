"""One bounded anonymous official HTTPS range; never loads a model or prints signed URLs."""
import hashlib,json,os,re,resource,signal,ssl,stat,time,urllib.error,urllib.parse,urllib.request
from common import D,P,RUN,need,save,load,own,budget,sha

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class GateError(ValueError):pass

def gate(ok,code):
    if not ok:raise GateError(code)

def url_fields(url):
    value=urllib.parse.urlsplit(url)
    gate(value.scheme=='https' and value.hostname in P['allowed_HTTPS_hosts'] and value.port in (None,443) and value.username is None and value.password is None and not value.fragment,'HTTPS_EXACT_HOST_NO_CREDENTIAL_URL')
    return dict(hostname=value.hostname,URL_SHA256=sha(url.encode()))

def main(context):
    begin=time.monotonic();epoch=context['epoch'];end=min(epoch+80,context['work_deadline']);left=budget(epoch,end-epoch)
    def expired(*_):raise TimeoutError('ORIGINAL_RANGE_WORK_DEADLINE')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,expired)
    signal.setitimer(signal.ITIMER_REAL,left())
    root=RUN/'worker';root.mkdir(mode=0o700);owner=own()
    result=dict(context,passed=False,owner=owner,HTTP_attempts=0,redirects=0,hops=[],explicit_body_consumed=0,body_written=0,expected_body_bytes=1048576,observed_prefix_SHA256=None,body_consumed_before_accepted_headers=0,physical_socket_prefetch='UNKNOWN_NOT_MEASURED',whole_SHA_verified=False,Model_GO=False,GPU_GO=False,training_GO=False,old362_queries=0,old_payload_reads_writes=0,A100_calls=0,Torch_SDK_GPU_calls=0,network_namespace='COMPUTE_NODE_ONLY',signed_URLs_saved=False,phase_costs={},error=None)
    response=None;file=None;prefix=root/'model03-bytes0-1048575.private';partial=prefix.with_suffix('.incomplete')
    def cost(name,fn):
        mark=time.monotonic()
        try:return fn()
        finally:
            rec=result['phase_costs'].setdefault(name,dict(seconds=0.,calls=0));rec['seconds']+=time.monotonic()-mark;rec['calls']+=1
    try:
        gate(owner['UID']==154459 and owner['host']==P['node'] and os.environ.get('SLURM_JOB_ID')==P['parent'] and os.environ.get('SLURM_STEP_ID','').isdigit(),'REAL_FIXED_COMPUTE_STEP')
        result['step']=P['parent']+'.'+os.environ['SLURM_STEP_ID']
        cpus=sorted(set(os.sched_getaffinity(0))-{0,1});gate(len(cpus)>=2,'TWO_WORKER_CPU_FROM_ALLOCATED4');os.sched_setaffinity(0,cpus[:2]);resource.setrlimit(resource.RLIMIT_AS,(P['AS_max_bytes'],P['AS_max_bytes']))
        nice=os.getpriority(os.PRIO_PROCESS,0)
        if nice<10:os.nice(10-nice)
        result.update(CPU_affinity=cpus[:2],allocated_CPU=4,worker_CPU=2,AS_limit_bytes=P['AS_max_bytes'],AS_is_not_RSS=True,nice=os.getpriority(os.PRIO_PROCESS,0))
        claim=load(D/'CLAIM.json');gate(claim['root_GO'] is claim['CPU_GO'] is claim['HTTP_GO'] is True,'PUBLISHED_CPU_HTTP_SCOPE_ONLY')
        save(root/'intent.json',dict(context,owner=owner,allocated_CPU=4,worker_CPU=2,URL_SHA256=sha(P['URL'].encode()),range_header=P['range_header'],body_cap=1048576))
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        current=P['URL'];headers=dict(Range='bytes=0-1048575',**{'Accept-Encoding':'identity','User-Agent':'bounded-internvl-model03-range-probe-v1'})
        while True:
            left();fields=url_fields(current);hop=dict(fields,HTTP_status=None,explicit_body_consumed=0);result['hops'].append(hop);result['HTTP_attempts']+=1
            gate(result['HTTP_attempts']<=4,'MAXIMUM_HTTP_ATTEMPTS')
            request=urllib.request.Request(current,headers=headers,method='GET')
            def request_once():
                try:return opener.open(request,timeout=min(20,left()))
                except urllib.error.HTTPError as error:return error
            response=cost('HTTPS_open_headers',request_once);status=response.status;hop['HTTP_status']=status
            if status in (301,302,303,307,308):
                locations=response.headers.get_all('Location',[]);gate(len(locations)==1 and locations[0],'UNIQUE_LOCATION_HEADER')
                next_url=urllib.parse.urljoin(current,locations[0]);hop.update(redirect_hostname=urllib.parse.urlsplit(next_url).hostname,redirect_URL_SHA256=sha(next_url.encode()));url_fields(next_url)
                response.close();response=None # No read of any redirect response body.
                result['redirects']+=1;gate(result['redirects']<=3,'MAXIMUM_REDIRECTS');current=next_url;continue
            gate(status==206,'FINAL_STATUS_MUST_BE206_REJECT_BODY')
            ranges=response.headers.get_all('Content-Range',[]);lengths=response.headers.get_all('Content-Length',[]);encodings=response.headers.get_all('Content-Encoding',[])
            hop.update(Content_Range_headers=len(ranges),Content_Length_headers=len(lengths),Content_Encoding_headers=len(encodings))
            gate(len(ranges)==1 and ranges[0].strip()=='bytes 0-1048575/4915962496','EXACT_CONTENT_RANGE_TOTAL_REQUIRED')
            gate(len(lengths)==1 and re.fullmatch(r'[0-9]+',lengths[0]) and int(lengths[0])==1048576,'EXACT_ONE_MIB_CONTENT_LENGTH_REQUIRED')
            gate(not response.headers.get_all('Transfer-Encoding',[]) and (not encodings or len(encodings)==1 and encodings[0].lower()=='identity'),'IDENTITY_NOT_CHUNKED_RESPONSE')
            result.update(accepted_headers=True,final_hostname=fields['hostname'],final_URL_SHA256=fields['URL_SHA256'],HTTP_status=206,Content_Range=ranges[0].strip(),Content_Length=int(lengths[0]))
            break
        digest=hashlib.sha256();file=partial.open('xb');opened=os.fstat(file.fileno())
        while result['explicit_body_consumed']<1048576:
            left();block=cost('body_read',lambda:response.read(min(65536,1048576-result['explicit_body_consumed'])));gate(block,'SHORT_RANGE_BODY')
            result['explicit_body_consumed']+=len(block);result['hops'][-1]['explicit_body_consumed']+=len(block);gate(result['explicit_body_consumed']<=1048576,'EXPLICIT_BODY_CAP')
            cost('body_hash',lambda:digest.update(block));written=cost('body_write',lambda:file.write(block));gate(written==len(block),'SHORT_PREFIX_WRITE');result['body_written']+=written
        file.flush();cost('prefix_fsync',lambda:os.fsync(file.fileno()));closed=os.fstat(file.fileno())
        gate(closed.st_size==1048576 and [opened.st_dev,opened.st_ino]==[closed.st_dev,closed.st_ino],'PREFIX_STABLE_SIZE_FD')
        file.close();file=None;response.close();response=None
        gate(not prefix.exists() and not prefix.is_symlink(),'NEW_PREFIX_EXISTS');partial.rename(prefix);final=prefix.stat()
        directory=os.open(prefix.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:cost('prefix_directory_fsync',lambda:os.fsync(directory))
        finally:os.close(directory)
        left()
        result.update(passed=True,observed_prefix_SHA256=digest.hexdigest(),prefix_file=prefix.name,prefix_stat=dict(st_dev=final.st_dev,st_ino=final.st_ino,st_size=final.st_size,st_mtime_ns=final.st_mtime_ns,st_ctime_ns=final.st_ctime_ns,st_uid=final.st_uid,st_mode=final.st_mode),body_consumed_before_accepted_headers=0)
    except BaseException as error:
        # Exception strings/tracebacks can contain signed redirect URLs; never log them.
        result['error']=dict(type=type(error).__name__,code=str(error) if type(error) is GateError else 'SANITIZED_EXTERNAL_ERROR')
    finally:
        if file is not None:
            try:file.close()
            except BaseException as error:result['passed']=False;result['error']=dict(type=type(error).__name__,code='SANITIZED_CLOSE_ERROR')
        if response is not None:
            try:response.close()
            except BaseException as error:result['passed']=False;result['error']=dict(type=type(error).__name__,code='SANITIZED_CLOSE_ERROR')
    result.update(finished_epoch=time.time(),worker_wall_seconds=time.monotonic()-begin,self_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,range_network_HTTP_only=True,full_weight_file_verified=False,wire_bytes='UNKNOWN_NOT_MEASURED',replay_allowed=False)
    save(root/'outcome.json',result);signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(passed=result['passed'],HTTP_attempts=result['HTTP_attempts'],redirects=result['redirects'],explicit_body_consumed=result['explicit_body_consumed'],whole_SHA_verified=False,signed_URLs_saved=False)))
    return 0 if result['passed'] else 1
