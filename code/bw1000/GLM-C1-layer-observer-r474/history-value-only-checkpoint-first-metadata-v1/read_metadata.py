"""First ten checkpoint metadata only. Never seek/read a tensor payload."""
import hashlib,json,math,os,re,resource,signal,stat,struct,time
from pathlib import Path
MODEL=Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3')
REGISTERED=(MODEL.parent,Path('/public/home/aclsylqidf/shared'))
CONFIG_SHA='3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69'
INDEX_SHA='e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf'
EXPECTED=[{'name': 'q_a_proj.weight', 'checkpoint_name': 'model.layers.0.self_attn.q_a_proj.weight', 'shape': [2048, 6144], 'safetensors_dtype': 'F8_E4M3', 'dtype_code': 2, 'bytes': 12582912, 'source_kind': 'original_checkpoint_range'}, {'name': 'q_a_proj.weight_scale_inv', 'checkpoint_name': 'model.layers.0.self_attn.q_a_proj.weight_scale_inv', 'shape': [16, 48], 'safetensors_dtype': 'F32', 'dtype_code': 0, 'bytes': 3072, 'source_kind': 'original_checkpoint_range'}, {'name': 'q_b_proj.weight', 'checkpoint_name': 'model.layers.0.self_attn.q_b_proj.weight', 'shape': [16384, 2048], 'safetensors_dtype': 'F8_E4M3', 'dtype_code': 2, 'bytes': 33554432, 'source_kind': 'original_checkpoint_range'}, {'name': 'q_b_proj.weight_scale_inv', 'checkpoint_name': 'model.layers.0.self_attn.q_b_proj.weight_scale_inv', 'shape': [128, 16], 'safetensors_dtype': 'F32', 'dtype_code': 0, 'bytes': 8192, 'source_kind': 'original_checkpoint_range'}, {'name': 'kv_a_proj_with_mqa.weight', 'checkpoint_name': 'model.layers.0.self_attn.kv_a_proj_with_mqa.weight', 'shape': [576, 6144], 'safetensors_dtype': 'F8_E4M3', 'dtype_code': 2, 'bytes': 3538944, 'source_kind': 'original_checkpoint_range'}, {'name': 'kv_a_proj_with_mqa.weight_scale_inv', 'checkpoint_name': 'model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv', 'shape': [5, 48], 'safetensors_dtype': 'F32', 'dtype_code': 0, 'bytes': 960, 'source_kind': 'original_checkpoint_range'}, {'name': 'kv_b_proj.weight', 'checkpoint_name': 'model.layers.0.self_attn.kv_b_proj.weight', 'shape': [28672, 512], 'safetensors_dtype': 'F8_E4M3', 'dtype_code': 2, 'bytes': 14680064, 'source_kind': 'original_checkpoint_range'}, {'name': 'kv_b_proj.weight_scale_inv', 'checkpoint_name': 'model.layers.0.self_attn.kv_b_proj.weight_scale_inv', 'shape': [224, 4], 'safetensors_dtype': 'F32', 'dtype_code': 0, 'bytes': 3584, 'source_kind': 'original_checkpoint_range'}, {'name': 'q_a_layernorm.weight', 'checkpoint_name': 'model.layers.0.self_attn.q_a_layernorm.weight', 'shape': [2048], 'safetensors_dtype': 'BF16', 'dtype_code': 1, 'bytes': 4096, 'source_kind': 'original_checkpoint_range'}, {'name': 'kv_a_layernorm.weight', 'checkpoint_name': 'model.layers.0.self_attn.kv_a_layernorm.weight', 'shape': [512], 'safetensors_dtype': 'BF16', 'dtype_code': 1, 'bytes': 1024, 'source_kind': 'original_checkpoint_range'}]
H=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,code):
    if not ok:raise ValueError(code)
def identity(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def strict(raw):
    def pairs(items):
        q={}
        for k,v in items:need(k not in q,'DUPLICATE_JSON_KEY');q[k]=v
        return q
    def bad(x):raise ValueError('NONFINITE_JSON')
    def real(x):
        v=float(x);need(math.isfinite(v),'NONFINITE_JSON');return v
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad,parse_float=real)
def main():
    need(globals().get('ROOT_APPROVED') is True,'ROOT_APPROVAL_REQUIRED')
    need(re.fullmatch('[0-9a-f]{32}',REQUEST),'REQUEST_ID')
    start=time.monotonic();allowance=EPOCH+90-time.time();need(0<allowance<=90,'ORIGIN_INVALID')
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));cores=sorted(os.sched_getaffinity(0));need(cores,'AFFINITY');os.sched_setaffinity(0,set(cores[:2]))
    def alarm(*unused):raise TimeoutError('ORIGIN_EXHAUSTED')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,allowance)
    q=dict(schema='GLM474-ten-required-first-checkpoint-metadata-v1',request_id=REQUEST,epoch=EPOCH,deadline=EPOCH+90,passed=False,metadata_only=True,planned_parameters=10,completed_parameters=0,rows=[],files={},registered_roots={},errors=[],header_bytes_read=0,prefix_bytes_read=0,config_index_bytes_read=0,payload_bytes_read=0,full_shard_hashes=0,actual_payload_range_SHA256=None,expected_future_payload_bytes=64377280,Native_GO=False,GPU_GO=False,Model_GO=False,replay_allowed=False,remote_writes=0,CPU_affinity=sorted(os.sched_getaffinity(0)),AS_hard_limit_bytes=512*1024**2)
    roots={};model_resolved=None;model_stat=None
    def left():need(min(EPOCH+90-time.time(),allowance-(time.monotonic()-start))>1,'ORIGIN_EXHAUSTED')
    def anchors():
        left()
        for logical,(resolved,devino) in roots.items():
            p=Path(logical);need(p.resolve(strict=True)==resolved,'REGISTERED_ROOT_RESOLVE_CHANGED');s=resolved.stat();need(stat.S_ISDIR(s.st_mode) and (s.st_dev,s.st_ino)==devino,'REGISTERED_ROOT_IDENTITY_CHANGED')
        need(MODEL.resolve(strict=True)==model_resolved,'MODEL_RESOLVE_CHANGED');s=model_resolved.stat();need(stat.S_ISDIR(s.st_mode) and (s.st_dev,s.st_ino)==model_stat,'MODEL_IDENTITY_CHANGED')
    def resolve_file(name,is_shard=False):
        anchors();need(Path(name).name==name and name not in ('','.','..'),'FIXED_SUFFIX')
        logical=MODEL/name;resolved=logical.resolve(strict=True)
        selected=[logical_root for logical_root,(base,unused) in roots.items() if resolved.is_relative_to(base)]
        need(selected,'OUTSIDE_REGISTERED_STORAGE')
        # Only the fixed logical file chosen by pinned metadata is allowed.
        # Its stable target may be a legitimate cache blob in registered storage;
        # canonical blob basenames need not equal the logical shard filename.
        need(logical.parent==MODEL and logical.name==name,'PINNED_LOGICAL_FILE_SUFFIX')
        s=resolved.stat();need(stat.S_ISREG(s.st_mode),'REGULAR_FILE_REQUIRED')
        return logical,resolved,s,selected
    def finish_file(logical,resolved,before,opened,after):
        need(identity(before)==identity(opened)==identity(after)==identity(resolved.stat()),'STABLE_FD')
        need(logical.resolve(strict=True)==resolved,'FILE_RESOLVE_CHANGED');anchors()
    def small(name,cap,pin):
        logical,resolved,s,selected=resolve_file(name);need(0<=s.st_size<=cap,'METADATA_SIZE_BOUND')
        with resolved.open('rb') as f:
            a=os.fstat(f.fileno());raw=f.read(cap+1);z=os.fstat(f.fileno())
        finish_file(logical,resolved,s,a,z);need(len(raw)==s.st_size and len(raw)<=cap and H(raw)==pin,'ORIGINAL_METADATA_SHA')
        q['config_index_bytes_read']+=len(raw);q['files'][name]=dict(resolved_path=str(resolved),registered_roots=selected,bytes=len(raw),SHA256=pin)
        return strict(raw)
    try:
        need(len(EXPECTED)==10 and len({r['checkpoint_name'] for r in EXPECTED})==10 and sum(r['bytes'] for r in EXPECTED)==64377280,'EXPECTED_TEMPLATE')
        for logical in REGISTERED:
            try:
                resolved=logical.resolve(strict=True);s=resolved.stat();need(stat.S_ISDIR(s.st_mode),'STORAGE_ROOT_NOT_DIRECTORY')
                roots[str(logical)]=(resolved,(s.st_dev,s.st_ino));q['registered_roots'][str(logical)]=dict(state='RESOLVED',resolved_path=str(resolved))
            except (OSError,ValueError) as e:q['registered_roots'][str(logical)]=dict(state='UNAVAILABLE',error_type=type(e).__name__)
        need(roots,'NO_REGISTERED_STORAGE_ROOT')
        model_resolved=MODEL.resolve(strict=True);s=model_resolved.stat();need(stat.S_ISDIR(s.st_mode) and any(model_resolved.is_relative_to(r[0]) for r in roots.values()),'MODEL_OUTSIDE_REGISTERED_STORAGE');model_stat=(s.st_dev,s.st_ino);q['actual_model_root']=str(model_resolved)
        small('config.json',1024**2,CONFIG_SHA)
        index=small('model.safetensors.index.json',32*1024**2,INDEX_SHA);mapping=index.get('weight_map');need(isinstance(mapping,dict),'WEIGHT_MAP_REQUIRED')
        selected={};grouped={}
        for row in EXPECTED:
            key=row['checkpoint_name'];need(key in mapping,'EXACT_REQUIRED_KEY_MISSING');name=mapping[key]
            need(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors',name),'INDEX_SHARD_FILENAME')
            selected[key]=name;grouped.setdefault(name,[]).append(row)
        need(1<=len(grouped)<=10,'SHARD_COUNT_BOUND');q['exact_mapping']=selected;q['planned_shards']=len(grouped)
        del index,mapping
        for name,rows in grouped.items():
            try:
                logical,resolved,s,registered=resolve_file(name,True)
                with resolved.open('rb') as f:
                    before=os.fstat(f.fileno());need(identity(before)==identity(s),'SHARD_OPEN_IDENTITY')
                    prefix=f.read(8);q['prefix_bytes_read']+=len(prefix);need(len(prefix)==8,'HEADER_PREFIX')
                    n=struct.unpack('<Q',prefix)[0]
                    need(0<n<=16*1024**2 and 8+n<=s.st_size and q['header_bytes_read']+n<=64*1024**2,'HEADER_BOUND')
                    raw=f.read(n);q['header_bytes_read']+=len(raw);need(len(raw)==n,'SHORT_HEADER');after=os.fstat(f.fileno())
                    need(f.tell()==8+n,'NO_PAYLOAD_READ')
                finish_file(logical,resolved,s,before,after);header=strict(raw);need(isinstance(header,dict),'HEADER_OBJECT')
                header_sha=H(raw);q['files'][name]=dict(resolved_path=str(resolved),registered_roots=registered,shard_bytes=s.st_size,header_bytes=n,header_SHA256=header_sha,header_prefix_SHA256=H(prefix),shard_payload_SHA256=None)
                ranges=[];verified=[]
                for expected in rows:
                    key=expected['checkpoint_name'];need(key in header and isinstance(header[key],dict),'SELECTED_HEADER_KEY_MISSING');t=header[key];shape=t.get('shape');offset=t.get('data_offsets')
                    need(isinstance(shape,list) and all(type(v)is int and v>0 for v in shape) and shape==expected['shape'] and t.get('dtype')==expected['safetensors_dtype'],'SELECTED_DTYPE_SHAPE_MISMATCH')
                    need(isinstance(offset,list) and len(offset)==2 and all(type(v)is int for v in offset),'SELECTED_OFFSET_SCHEMA');begin,end=offset
                    need(0<=begin<end<=s.st_size-8-n and end-begin==expected['bytes'],'SELECTED_OFFSET_BYTES_BOUND');ranges.append((begin,end))
                    verified.append(dict(expected,shard=name,resolved_shard=str(resolved),shard_bytes=s.st_size,header_bytes=n,header_SHA256=header_sha,data_offsets=offset,absolute_offset=8+n+begin,range_SHA256=None,metadata_passed=True,payload_qualified=False))
                ranges.sort();need(all(ranges[i][1]<=ranges[i+1][0] for i in range(len(ranges)-1)),'SELECTED_RANGES_OVERLAP')
                q['rows'].extend(verified);q['completed_parameters']+=len(verified)
                del header,raw
            except (OSError,ValueError,KeyError,UnicodeError,struct.error) as e:
                q['errors'].append(dict(shard=name,required_keys=[r['checkpoint_name'] for r in rows],error_type=type(e).__name__,code=str(e)[:100] if isinstance(e,ValueError) else 'READ_OR_SCHEMA_FAILURE'))
        need(q['completed_parameters']==10 and not q['errors'],'TEN_PARAMETER_METADATA_INCOMPLETE');q['passed']=True
    except BaseException as e:q['errors'].append(dict(stage='admission_or_budget',error_type=type(e).__name__,code=str(e)[:100] if isinstance(e,ValueError) else 'READ_OR_BUDGET_FAILURE'))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);q['missing_parameters']=10-q['completed_parameters'];q['elapsed_s']=time.monotonic()-start;q['total_read_bytes']=q['config_index_bytes_read']+q['prefix_bytes_read']+q['header_bytes_read'];q['reader_process_peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        need(len(json.dumps(q).encode())<=65536,'OUTPUT_BOUND');print(json.dumps(q,allow_nan=False),flush=True)
if __name__=='__main__':main()
