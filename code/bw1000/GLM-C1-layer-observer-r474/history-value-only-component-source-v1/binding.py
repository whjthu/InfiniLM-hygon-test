"""Future explicit GPU component binding; no default entry and no Torch dependency."""
import ctypes as ct,hashlib,json,re,struct
from pathlib import Path
from controlled_inputs import case
H=lambda b:hashlib.sha256(b).hexdigest()
MODEL=Path('/old/models/ZhipuAI-GLM-5.3')
def need(ok,code):
    if not ok:raise ValueError(code)
def read_parameter(row):
    need(row['source_kind']=='original_checkpoint_range' and re.fullmatch('[0-9a-f]{64}',row.get('range_SHA256') or ''),'UNQUALIFIED_CHECKPOINT_RANGE')
    name=row['shard'];need(Path(name).name==name and name.endswith('.safetensors'),'CHECKPOINT_FILENAME')
    p=MODEL/name;s=p.stat();need(s.st_size==row['shard_bytes'],'CHECKPOINT_BYTES')
    identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    import os
    with p.open('rb') as f:
        before=os.fstat(f.fileno());need(identity(before)==identity(s),'CHECKPOINT_OPEN_CHANGED')
        n=struct.unpack('<Q',f.read(8))[0];need(n==row['header_bytes'] and 0<n<=16*1024**2,'CHECKPOINT_HEADER_BOUND')
        raw=f.read(n);need(len(raw)==n and H(raw)==row['header_SHA256'],'CHECKPOINT_HEADER_PIN');header=json.loads(raw)
        info=header[row['checkpoint_name']];need(info['shape']==row['shape'] and info['dtype']==row['safetensors_dtype'],'CHECKPOINT_TENSOR_METADATA')
        start,end=info['data_offsets'];offset=8+n+start
        need(offset==row['absolute_offset'] and end-start==row['bytes'] and 0<=start<end<=s.st_size-8-n and row['bytes']<=64*1024**2,'CHECKPOINT_RANGE_BOUND')
        f.seek(offset);value=f.read(row['bytes']);need(len(value)==row['bytes'] and H(value)==row['range_SHA256'],'CHECKPOINT_RANGE_SHA')
        need(identity(before)==identity(os.fstat(f.fileno()))==identity(p.stat()),'CHECKPOINT_STABLE_FD')
    return value
def run_component(manifest,input_pins):
    # All missing actual metadata is a hard prerequisite, before Native dlopen.
    need(manifest.get('GPU_GO') is True and manifest.get('component_CPU_qualified') is True and manifest.get('checkpoint_ranges_qualified') is True,'COMPONENT_NOT_ADOPTED')
    need(manifest['model_path']==str(MODEL) and manifest['TP']==1 and manifest['heads']==64,'REAL_TP1_SCOPE')
    need(len(manifest['parameters'])==10 and len({r['name'] for r in manifest['parameters']})==10,'EXACT10_PARAMETERS')
    for row in manifest['parameters']:need(re.fullmatch('[0-9a-f]{64}',row.get('range_SHA256') or ''),'MISSING_ACTUAL_RANGE_PIN')
    cfg=(MODEL/'config.json').read_bytes();index=(MODEL/'model.safetensors.index.json').read_bytes()
    need(H(cfg)==manifest['config_SHA256'] and H(index)==manifest['index_SHA256'],'ORIGINAL_CHECKPOINT_METADATA')
    mapping=json.loads(index)['weight_map']
    for row in manifest['parameters']:need(mapping[row['checkpoint_name']]==row['shard'],'ACTUAL_INDEX_MAPPING')
    libpath=Path(manifest['component_dso']);need(re.fullmatch('[0-9a-f]{64}',manifest.get('component_dso_SHA256') or '') and H(libpath.read_bytes())==manifest['component_dso_SHA256'],'NEW_HOST_DSO_PIN')
    # The one wrapper links only179 B, whose original expand_cached math is unchanged.
    parent=Path('/refactor/results/GLM474-history-value-only-common-AB-CPU-v1/B/libGLM_history_value_B.so')
    need(H(parent.read_bytes())=='02b0b1e8cf529fcb73ad6bd429c78d158beb6d19d4577866977abe97f686136d','PARENT179_B_PIN')
    lib=ct.CDLL(str(libpath),mode=ct.RTLD_LOCAL)
    lib.glm_history_component_create.argtypes=[ct.c_char_p];lib.glm_history_component_create.restype=ct.c_void_p
    for name in ('keys','result','error'):
        f=getattr(lib,'glm_history_component_'+name);f.argtypes=[ct.c_void_p] if name=='keys' else [];f.restype=ct.c_char_p
    lib.glm_history_component_load.argtypes=[ct.c_void_p,ct.c_char_p,ct.c_void_p,ct.c_size_t,ct.c_int];lib.glm_history_component_load.restype=ct.c_int
    lib.glm_history_component_finish.argtypes=[ct.c_void_p];lib.glm_history_component_finish.restype=ct.c_int
    lib.glm_history_component_pair.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,ct.c_int];lib.glm_history_component_pair.restype=ct.c_int
    lib.glm_history_component_close.argtypes=[ct.c_void_p];lib.glm_history_component_close.restype=ct.c_int
    lib.glm_history_component_guards.argtypes=[ct.c_void_p];lib.glm_history_component_guards.restype=ct.c_int
    ctx=lib.glm_history_component_create(cfg);need(ctx,'REAL_COMPONENT_FACTORY_FAILED')
    report=dict(passed=False,planned_pairs=3,completed_pairs=0,planned_warm_pairs=2,completed_warm_pairs=0,pairs=[],normal_close=False,Model_GO=False,TP1_not_TP16=True)
    try:
        rows=json.loads(lib.glm_history_component_keys(ctx));actual={r['name']:r for r in rows}
        need(set(actual)=={r['name'] for r in manifest['parameters']},'ACTUAL_REQUIRED_KEYS')
        for row in manifest['parameters']:
            need(all(actual[row['name']][k]==row[k] for k in ('shape','bytes','dtype_code')),'ACTUAL_FACTORY_PARAMETER_GEOMETRY')
            raw=read_parameter(row);owner=ct.create_string_buffer(raw)
            need(lib.glm_history_component_load(ctx,row['name'].encode(),owner,len(raw),row['dtype_code'])==0,'REAL_PARAMETER_LOAD_FAILED')
        need(lib.glm_history_component_finish(ctx)==0,'ORIGINAL_POSTLOAD_FAILED')
        # Two declared warm pairs; every timed pair remains in the denominator.
        for ordinal,first,warm in [(0,0,True),(0,1,True),(0,0,False),(1,1,False),(2,0,False)]:
            data=case(ordinal);pins=input_pins['cases'][ordinal]
            for k in ('latent','rotary','ids'):need(len(data[k])==pins['files'][k]['bytes'] and H(data[k])==pins['files'][k]['SHA256'],'CONTROLLED_INPUT_BYTES')
            owners=[ct.create_string_buffer(data[k]) for k in ('latent','rotary','ids')]
            need(lib.glm_history_component_pair(ctx,owners[0],len(data['latent']),owners[1],len(data['rotary']),owners[2],data['U'],first)==0,'PAIRED_V_OR_LIFETIME_FAILED')
            result=json.loads(lib.glm_history_component_result());result.update(input_ordinal=ordinal,warm=warm)
            report['pairs'].append(result);report['completed_warm_pairs' if warm else 'completed_pairs']+=1
        need(lib.glm_history_component_guards(ctx)==0,'NATIVE_GUARD_MISMATCH');report['guards']=json.loads(lib.glm_history_component_result())
        report['passed']=report['completed_pairs']==3 and report['completed_warm_pairs']==2 and report['guards']['actual_rejections']==10
    except BaseException as e:
        report['error_type']=type(e).__name__;report['error_SHA256']=H(str(e).encode());report['passed']=False
    finally:
        close=lib.glm_history_component_close(ctx);report['normal_close']=close==0
        if close==0:report['close']=json.loads(lib.glm_history_component_result())
        report['passed']=report['passed'] and report['normal_close']
    return report
