"""Once HOST stdlib: seven saved texts, frozen producer metadata, two pinned headers."""
import base64,hashlib,json,math,os,re,resource,signal,stat,time
from pathlib import Path
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
W=R/'results/GLM474-history-value-only-common-AB-CPU-v1'
OLD=R/'results/GLM474-history-value-only-common-AB-CPU-reader-v1'
H=lambda b:hashlib.sha256(b).hexdigest()
OUTPUTS={'reference-independent-nm.stdout': {'bytes': 188070, 'SHA256': '5ac2a9f07eac819bebf199ef236690ef1a29fb92f6d6e012158681efbf6f780a'}, 'A-independent-nm.stdout': {'bytes': 188070, 'SHA256': 'b90b322112e8fbc9560d922488d368c7d5ebc0900736ce14d3c98673387de378'}, 'A-independent-dynamic.stdout': {'bytes': 2599, 'SHA256': '11b82419e467216ccb52f28c77437a2d1ed0d51562b937a480dbe09c263939d8'}, 'A-independent-tls.stdout': {'bytes': 1248046, 'SHA256': 'b7a0955198b51cccab83915f34ada81a9c1edfaa76185ef4dd347ff97c8fe359'}, 'B-independent-nm.stdout': {'bytes': 188382, 'SHA256': 'd96cd44b626acca5610cf1663d32edfb32bb9fb7f39f8c043af43aa759d37e6f'}, 'B-independent-dynamic.stdout': {'bytes': 2599, 'SHA256': '6a916de87a2951dc3b73e7064ea9f7d1a1f85cb1f2619e5aa81ae8c80638839e'}, 'B-independent-tls.stdout': {'bytes': 1249546, 'SHA256': 'e5255c7d6f91d77bbbd5667523543d6a307c200f1592eed10ec4f32ff9da6aa6'}}
META={
 'outcome.json':(1374052,'72558b96b8c7b3ef39d61431c403ffd0dce06116e608504c1028b29fbd58bddf'),
 'build-result.json':(1350034,'6b098aaf201f1a45f64f4b8d68c40c5b32ac1ed4cfc2e8ea72ec283b568e1560'),
 'src/PLAN.json':(1450080,'d5356d5b2bc60825537d544ee9f1d100e659f0569eb2521177a466acc7a82fea'),
 'src/build_once.py':(15568,'30f8f2586b2fdc99eeed7f829af22fff92e617f613b3117c9af402fe978cf195')}
HEADERS={
 'communication_group.hpp':'1844035e0f24eb929781eb63e86a25b2de8e89852a77c5cfa9c1644a5f62c16f',
 'dist_config.hpp':'5b6c249e946d972f4b523ef17187025483c5da2e38a830812ee2e8e8055e35b6'}
HEADER_DIR=W/'B/project-mirror/candidates/rebased-5f91ad7-r3668/Model/GLM/InfiniLM/csrc/engine/distributed'
def need(ok,code):
    if not ok:raise ValueError(code)
def strict(raw):
    def pairs(items):
        q={}
        for k,v in items:need(k not in q,'DUPLICATE_JSON');q[k]=v
        return q
    def bad(x):raise ValueError('NONFINITE_JSON')
    def real(x):
        v=float(x);need(math.isfinite(v),'NONFINITE_JSON');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)
def main():
    need(globals().get('ROOT_APPROVED') is True,'ROOT_APPROVAL_REQUIRED')
    need(re.fullmatch('[0-9a-f]{32}',REQUEST),'REQUEST_ID')
    start=time.monotonic();remaining=EPOCH+60-time.time();need(0<remaining<=60,'ORIGIN')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));cores=sorted(os.sched_getaffinity(0));need(cores,'AFFINITY');os.sched_setaffinity(0,set(cores[:2]))
    def alarm(*unused):raise TimeoutError('ORIGIN_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,remaining)
    q=dict(schema='GLM474-static-ABI-two-source-headers-v1',request_id=REQUEST,epoch=EPOCH,deadline=EPOCH+60,passed=False,static_ABI_passed=False,provider_record_passed=False,first_source_headers_passed=False,files={},headers={},failures=[],total_read_bytes=0,remote_writes=0,tool_executions=0,container_entries=0,new_provider_body_hashes=0,prior_root_provider_body_hashes=32,GPU_GO=False,Model_GO=False,component_GO=False,replay_allowed=False,prior_reader_false_preserved=True,remaining39_timeout_preserved=True,original_client_timeout_preserved=True)
    def read(p,cap,pin,size=None):
        need(min(EPOCH+60-time.time(),remaining-(time.monotonic()-start))>1,'ORIGIN_EXHAUSTED')
        p=Path(p);s=p.stat();need(stat.S_ISREG(s.st_mode) and 0<=s.st_size<=cap,'FILE_BOUND')
        need(size is None or s.st_size==size,'FILE_BYTES')
        with p.open('rb') as f:
            a=os.fstat(f.fileno());raw=f.read(cap+1);b=os.fstat(f.fileno())
        ident=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        need(len(raw)<=cap and ident(s)==ident(a)==ident(b)==ident(p.stat()),'STABLE_FD')
        q['total_read_bytes']+=len(raw);need(q['total_read_bytes']<=8*1024**2,'AGGREGATE8MIB')
        sha=H(raw);need(sha==pin,'FILE_SHA');q['files'][str(p)]=dict(bytes=len(raw),SHA256=sha);return raw
    try:
        meta={}
        for name,(size,sha) in META.items():
            raw=read(W/name,2*1024**2,sha,size)
            if name.endswith('.json'):meta[name]=strict(raw)
        out=meta['outcome.json'];build=meta['build-result.json'];plan=meta['src/PLAN.json']
        need(out['passed'] is True and out['rc']==0 and out['build_SHA256']==META['build-result.json'][1] and out['build']==build,'ORIGINAL_COMPLETED_BUILD')
        row=out['accounting']['row'];need(row[0]=='1103606.179' and row[1]==W.name and row[2:4]==['COMPLETED','0:0'] and row[5]=='f13r1n05','EXACT_TERMINAL')
        need(out['epoch']==1794933214.2382555 and out['deadline']==1794935014.2382555 and build['step']==row[0] and build['passed'] is True,'ORIGINAL_EPOCH_STEP')
        need(build['actual']==dict(freshM=64,HOST_compile=50,GPU_compile=14,objects=64,links=2),'ORIGINAL_DENOMINATORS')
        original=plan['original169'];providers=original['providers']
        need(len(providers)==71 and H(json.dumps(providers,sort_keys=True,separators=(',',':')).encode())=='b433fe02fc229c6a2bb3768509f49155a68b60f0860fb3b2325c01a2174c8d4a','PROVIDER_MAP71')
        cmds={}
        for c in build['commands']:need(c['label'] not in cmds,'DUPLICATE_COMMAND');cmds[c['label']]=c
        for label in ('reference-ldd','A-ldd','B-ldd'):
            c=cmds[label];need(c['rc']==0 and c['owned_child_reaped'] is True and c['argv'][:2]==['/usr/bin/ldd','-r'] and re.fullmatch('[0-9a-f]{64}',c['stdout_SHA256']) and re.fullmatch('[0-9a-f]{64}',c['stderr_SHA256']),'ORIGINAL_PROVIDER_CHECK_COMMAND')
        q['provider_record_passed']=True
        q['provider_evidence']=dict(count=71,expected_map_SHA256='b433fe02fc229c6a2bb3768509f49155a68b60f0860fb3b2325c01a2174c8d4a',provenance='Original179 frozen build command completed providers() hashing each file and equality to pinned PLAN; raw per-file observed hash map was not separately persisted. This is producer source-contract/terminal evidence, not71 new independent body hashes.',body_hashes_by_this_reader=0)
        texts={name:read(OLD/name,2*1024**2,pin['SHA256'],pin['bytes']).decode() for name,pin in OUTPUTS.items()}
        need(len(texts)==7,'SEVEN_TEXTS')
        def strong(text):
            found={}
            for line in text.splitlines():
                r=line.split()
                if len(r)>=3 and r[-2] in ('T','D','B','R'):need(r[-1] not in found,'DUPLICATE_SYMBOL');found[r[-1]]=r[-2]
            return found
        reference=strong(texts['reference-independent-nm.stdout']);need(len(reference)==280,'REFERENCE280')
        q['libraries']={}
        for arm,expected_sha,expected_bytes in [('A','e2486277c55b1b3814d6ce484eec63c0163a635d6204adaf30b363638b9eb7c0',1930288),('B','02b0b1e8cf529fcb73ad6bd429c78d158beb6d19d4577866977abe97f686136d',1934848)]:
            record=build['libraries'][arm];need(record['SHA256']==expected_sha and record['bytes']==expected_bytes and record['providers']==71,'LIBRARY_METADATA')
            syms=strong(texts[arm+'-independent-nm.stdout']);need(all(syms.get(k)==v for k,v in reference.items()),'STRONG_BASELINE_LOST');delta=set(syms)-set(reference)
            need(not delta if arm=='A' else len(delta)==1 and all('AttentionPrepare' in n and 'expand_cached_values' in n for n in delta),'EXACT_CPP_DELTA')
            need(record['strong_exports']==len(syms) and record['new_strong']==sorted(delta),'PRODUCER_SYMBOL_RECORD')
            dynamic=texts[arm+'-independent-dynamic.stdout'];needed=re.findall(r'\(NEEDED\).*Shared library: \[([^]]+)\]',dynamic)
            need('SYMBOLIC' in dynamic and needed==original['NEEDED'] and len(needed)==14,'NEEDED14_SYMBOLIC')
            for name in ('glm_selected_value_layer0::detail::decode_owner','glm_sparse_r131::current'):
                rows=[s.split() for s in texts[arm+'-independent-tls.stdout'].splitlines() if s.rstrip().endswith(' '+name)]
                need(len({(r[1],r[6]) for r in rows if len(r)>6 and r[3]=='TLS' and r[6]!='UND'})==1,'TLS_SINGLETON')
            q['libraries'][arm]=dict(reused_DSO_SHA256=expected_sha,reused_DSO_bytes=expected_bytes,strong=len(syms),added=sorted(delta),NEEDED=14,TLS=2)
        q['static_ABI_passed']=True
    except BaseException as e:q['failures'].append(dict(stage='static_ABI_provider_record',error_type=type(e).__name__,code=str(e)[:100] if isinstance(e,ValueError) else 'IO_OR_BUDGET'))
    # The first exact source-body fetch is distinct from the completed ABI evidence.
    for name,pin in HEADERS.items():
        try:
            raw=read(HEADER_DIR/name,65536,pin);q['headers'][name]=dict(bytes=len(raw),SHA256=pin,source_b64=base64.b64encode(raw).decode())
        except BaseException as e:q['failures'].append(dict(stage='source_header:'+name,error_type=type(e).__name__))
    signal.setitimer(signal.ITIMER_REAL,0);q['first_source_headers_passed']=len(q['headers'])==2;q['passed']=q['static_ABI_passed'] and q['provider_record_passed'] and q['first_source_headers_passed'];q['elapsed_s']=time.monotonic()-start
    need(len(json.dumps(q).encode())<=262144,'OUTPUT_BOUND');print(json.dumps(q,allow_nan=False),flush=True)
if __name__=='__main__':main()
