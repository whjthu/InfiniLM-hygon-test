"""First exact ten GLM checkpoint range hashes; stdlib HOST only, no raw export."""
import base64,hashlib,json,math,os,re,resource,signal,stat,struct,time
from pathlib import Path
META_SHA='15535b1e7da234c04fa7a2accdee379ed4233c7f7265795362447d3dfacf7b08'
META_REQUEST='2872317dceb34c2d8348a8e5e3f10fbe'
MODEL=Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3')
ROOTS=(MODEL.parent,Path('/public/home/aclsylqidf/shared'))
SHARD='model-00001-of-00141.safetensors'
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def ident(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'DUPLICATE_KEY');out[k]=v
        return out
    def bad(x):raise ValueError('NONFINITE_JSON')
    def real(x):
        v=float(x);need(math.isfinite(v),'NONFINITE_JSON');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)
def main():
    need(globals().get('ROOT_APPROVED') is True,'ROOT_APPROVAL_REQUIRED')
    need(re.fullmatch('[0-9a-f]{32}',REQUEST) and re.fullmatch('[0-9a-f]{40}',PUBLICATION),'IDENTIFIERS')
    started=time.monotonic();remaining=EPOCH+90-time.time();need(math.isfinite(remaining) and 0<remaining<=90,'ORIGINAL_EPOCH')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    cores=sorted(c for c in os.sched_getaffinity(0) if c not in (0,1));need(len(cores)>=2,'TWO_CPU_EXCLUDING01_REQUIRED');os.sched_setaffinity(0,set(cores[:2]))
    def alarm(*unused):raise TimeoutError('ORIGINAL_EPOCH_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,remaining)
    q=dict(schema='GLM474-first-ten-checkpoint-ranges-v1',request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+90,passed=False,metadata_SHA256=META_SHA,metadata_request=META_REQUEST,planned_parameters=10,attempted_parameters=0,completed_parameters=0,rows=[],errors=[],payload_bytes_read=0,header_bytes_read=0,prefix_bytes_read=0,index_config_bytes_read=0,full_shard_hashes=0,raw_weight_export_bytes=0,remote_writes=0,Native_GO=False,GPU_GO=False,Model_GO=False,replay_allowed=False,AS_hard_limit_bytes=512*1024**2,CPU_affinity=sorted(os.sched_getaffinity(0)))
    def left():need(min(EPOCH+90-time.time(),remaining-(time.monotonic()-started))>1,'ORIGINAL_EPOCH_EXHAUSTED')
    try:
        need(isinstance(METADATA_B64,str) and len(METADATA_B64)<=16384,'METADATA_TRANSPORT_BOUND')
        rawmeta=base64.b64decode(METADATA_B64,validate=True);need(len(rawmeta)==10059 and H(rawmeta)==META_SHA,'ORIGINAL_METADATA_PIN');m=strict(rawmeta)
        need(m['passed'] is True and m['metadata_only'] is True and m['request_id']==META_REQUEST and m['epoch']==1794939440.3945646 and m['completed_parameters']==10 and m['planned_shards']==1 and m['payload_bytes_read']==0 and not m['errors'],'ORIGINAL_METADATA_QUALIFICATION')
        rows=m['rows'];need(len(rows)==10 and len({r['checkpoint_name'] for r in rows})==10 and sum(r['bytes'] for r in rows)==64377280,'TEN_EXACT_RANGES')
        need({r['shard'] for r in rows}=={SHARD} and m['exact_mapping']=={r['checkpoint_name']:SHARD for r in rows},'EXACT_MAPPING')
        pin=m['files'][SHARD];need(pin['header_bytes']==21840 and pin['shard_bytes']==5363940952,'ORIGINAL_SHARD_GEOMETRY')
        registered={}
        for logical in ROOTS:
            resolved=logical.resolve(strict=True);s=resolved.stat()
            need(stat.S_ISDIR(s.st_mode) and str(resolved)==m['registered_roots'][str(logical)]['resolved_path'],'REGISTERED_ROOT_CHANGED')
            registered[logical]=(resolved,(s.st_dev,s.st_ino))
        model=MODEL.resolve(strict=True);ms=model.stat();need(str(model)==m['actual_model_root'] and any(model.is_relative_to(r[0]) for r in registered.values()),'MODEL_ROOT_CHANGED')
        model_id=(ms.st_dev,ms.st_ino);logical=MODEL/SHARD;resolved=logical.resolve(strict=True)
        need(str(resolved)==pin['resolved_path'] and all(r['resolved_shard']==str(resolved) for r in rows) and any(resolved.is_relative_to(r[0]) for r in registered.values()),'FIXED_LOGICAL_SHARD_RESOLVE')
        before=resolved.stat();need(stat.S_ISREG(before.st_mode) and before.st_size==pin['shard_bytes'],'SHARD_STAT')
        def anchors():
            left();need(logical.resolve(strict=True)==resolved and MODEL.resolve(strict=True)==model,'RESOLVE_CHANGED')
            s=model.stat();need((s.st_dev,s.st_ino)==model_id,'MODEL_IDENTITY_CHANGED')
            for root,(target,devino) in registered.items():
                s=target.stat();need(root.resolve(strict=True)==target and (s.st_dev,s.st_ino)==devino,'REGISTERED_ROOT_CHANGED')
        with resolved.open('rb',buffering=0) as f:
            opened=os.fstat(f.fileno());need(ident(opened)==ident(before),'OPEN_IDENTITY')
            left();prefix=f.read(8);q['prefix_bytes_read']=len(prefix)
            need(len(prefix)==8 and H(prefix)==pin['header_prefix_SHA256'] and struct.unpack('<Q',prefix)[0]==21840,'EXACT_HEADER_PREFIX')
            headerraw=f.read(21840);q['header_bytes_read']=len(headerraw);need(len(headerraw)==21840 and H(headerraw)==pin['header_SHA256'],'EXACT_HEADER_SHA');header=strict(headerraw)
            intervals=[]
            for row in rows:
                need(row['metadata_passed'] is True and row['payload_qualified'] is False and row['range_SHA256'] is None and row['source_kind']=='original_checkpoint_range','ORIGINAL_RANGE_STATUS')
                t=header[row['checkpoint_name']];start,end=row['data_offsets']
                need(t['shape']==row['shape'] and t['dtype']==row['safetensors_dtype'] and t['data_offsets']==row['data_offsets'],'CURRENT_HEADER_SELECTED_ROW')
                need(type(start)is int and type(end)is int and 0<=start<end<=before.st_size-21848 and end-start==row['bytes'] and row['absolute_offset']==21848+start,'EXACT_RANGE_GEOMETRY')
                intervals.append((start,end))
            intervals.sort();need(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),'SELECTED_RANGE_OVERLAP')
            # All ten metadata/header contracts are checked before the first payload byte.
            for row in rows:
                anchors();need(ident(os.fstat(f.fileno()))==ident(before),'BEFORE_RANGE_STABLE_FD');q['attempted_parameters']+=1
                f.seek(row['absolute_offset']);digest=hashlib.sha256();count=0
                while count<row['bytes']:
                    left();b=f.read(min(1024**2,row['bytes']-count));need(b,'SHORT_SELECTED_RANGE')
                    count+=len(b);q['payload_bytes_read']+=len(b);need(q['payload_bytes_read']<=64377280,'TOTAL_RANGE_BOUND');digest.update(b)
                need(f.tell()==row['absolute_offset']+row['bytes'] and ident(os.fstat(f.fileno()))==ident(before),'AFTER_RANGE_STABLE_FD')
                q['rows'].append(dict(name=row['name'],checkpoint_name=row['checkpoint_name'],shard=SHARD,resolved_shard=str(resolved),shape=row['shape'],safetensors_dtype=row['safetensors_dtype'],dtype_code=row['dtype_code'],data_offsets=row['data_offsets'],absolute_offset=row['absolute_offset'],bytes=count,header_SHA256=pin['header_SHA256'],range_SHA256=digest.hexdigest(),whole_read_stable=False))
            need(ident(os.fstat(f.fileno()))==ident(before),'FINAL_FD_IDENTITY')
        anchors();need(ident(resolved.stat())==ident(before),'FINAL_PATH_IDENTITY')
        need(len(q['rows'])==10 and q['payload_bytes_read']==64377280,'TEN_RANGE_COMPLETION')
        for row in q['rows']:row['whole_read_stable']=True
        q.update(passed=True,completed_parameters=10,shard=dict(logical_path=str(logical),resolved_path=str(resolved),bytes=before.st_size,identity=list(ident(before)),header_SHA256=pin['header_SHA256'],header_bytes=21840))
    except BaseException as error:
        q['errors'].append(dict(error_type=type(error).__name__,code=str(error)[:100] if isinstance(error,ValueError) else 'READ_OR_BUDGET_FAILURE'))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['missing_parameters']=10-q['completed_parameters'];q['elapsed_s']=time.monotonic()-started;q['reader_process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        q['total_checkpoint_bytes_read']=q['prefix_bytes_read']+q['header_bytes_read']+q['payload_bytes_read'];text=json.dumps(q,allow_nan=False);need(len(text.encode())<=32768,'OUTPUT_BOUND');print(text,flush=True)
if __name__=='__main__':main()
