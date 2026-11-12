"""One original77-file model and preset adapter full read; no tensor decode/export."""
from pathlib import Path
import datetime,hashlib,json,os,resource,signal,socket,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework'
epoch=float(os.environ['CONTRACT_CPU_EPOCH']);deadline=float(os.environ['CONTRACT_CPU_DEADLINE'])
start=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=540
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original complete file identity expired')))
signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
    with (W/n).open('x') as f:json.dump(v,f,indent=2)
def check():
    if min(deadline-time.time(),remaining-(time.monotonic()-start))<=0:raise TimeoutError('original single hash epoch expired')
def owner():return dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]))
events=[];phase='source';label='none'
def progress(v):
    e=dict(at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),component=label,**v);events.append(e)
    p=W/'hash-progress.tmp';p.write_text(json.dumps(e));p.replace(W/'hash-progress.json')
    with (W/'hash-progress.private.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
dump('CPU-owner.json',owner())
try:
    fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes())
    for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
    source=json.loads((S/'ORIGINAL_FILE_STATS.json').read_bytes())
    assert source['host']==socket.gethostname() and len(source['files'])==77
    root=Path(source['model_root']);adapter=source['adapter'];ap=Path(adapter['path'])
    sys.path.insert(0,str(F))
    from tools.hygon_indexed_model_identity import inspect_listed_files,stat_identity
    names=sorted(source['files']);keys=('bytes','mtime_ns','inode','device')
    model_stats={n:{k:source['files'][n][k] for k in keys} for n in names}
    adapter_stats={ap.name:{k:adapter[k] for k in keys}}
    assert stat_identity(ap)==adapter_stats[ap.name] and not ap.is_symlink()
    progress(dict(stage='whole_plan',model_files=names,preset_files=[ap.name],planned_files=78,
        planned_bytes=sum(v['bytes'] for v in model_stats.values())+adapter['bytes'],completed_files=[],bytes_read=0))
    phase='full-original-model77';label='model';hash_start=time.monotonic()
    model=inspect_listed_files(root,model_stats,check_deadline=check,progress=progress)
    assert model['completed_files']==77 and model['full_file_bytes_read']==32901268708
    phase='full-original-preset1';label='preset'
    preset=inspect_listed_files(ap.parent,adapter_stats,check_deadline=check,progress=progress)
    assert preset['completed_files']==1 and preset['full_file_bytes_read']==170128256
    binding=json.loads((S/'ORIGINAL_BINDING.json').read_bytes())
    assert preset['files'][ap.name]['SHA256']==binding['declared_preset_file_SHA256']
    for n,pin in binding['original_transformer_headers'].items():
        actual=model['files']['transformer/'+n]
        assert actual['header_bytes']==pin['header_bytes'] and actual['header_SHA256']==pin['header_SHA256']
    assert preset['files'][ap.name]['header_SHA256']==binding['original_preset_header']['header_SHA256']
    assert sum(len(model['files']['transformer/'+n]['tensors']) for n in binding['original_transformer_headers'])==521
    assert len(preset['files'][ap.name]['tensors'])==480
    check();assert 'torch' not in sys.modules
    identity=dict(model=model,preset=preset,model_root=str(root),preset_path=str(ap),
        original_inventory_SHA256=sha(S/'ORIGINAL_FILE_STATS.json'),original_binding_SHA256=sha(S/'ORIGINAL_BINDING.json'),
        observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),epoch=epoch,deadline=deadline,
        original_declared_model_identity=binding['declared_model_identity'],
        original_declared_model_identity_semantics='Declaration from original launch;not asserted equivalent to newly defined complete file-manifest digest.',
        declared_preset_full_SHA_matched=True,historical_training_file_bytes_qualified=False,official_release_provenance_qualified=False,
        actual_model_loading_qualified=False,actual_preset_fusion_qualified=False,GPU_GO=False,Model_GO=False)
    dump('MODEL_IDENTITY.json',identity)
    result=dict(passed=True,observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner(),epoch=epoch,deadline=deadline,
        framework_commit=fm['framework_commit'],framework_subset_files=len(fm['files']),model_files=77,preset_files=1,
        full_file_bytes_read=model['full_file_bytes_read']+preset['full_file_bytes_read'],
        model_full_file_bytes_read=model['full_file_bytes_read'],preset_full_file_bytes_read=preset['full_file_bytes_read'],
        model_weight_files=7,preset_weight_files=1,transformer_tensors=521,preset_tensors=480,
        MODEL_IDENTITY_SHA256=sha(W/'MODEL_IDENTITY.json'),progress_SHA256=sha(W/'hash-progress.private.jsonl'),progress_events=len(events),
        hash_wall_seconds=time.monotonic()-hash_start,wall_seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        decoded_tensor_values=False,weight_values_exported=False,model_loaded=False,Torch_imported=False,GPU_cards=0,GPU_GO=False,Model_GO=False,
        declared_preset_full_SHA_matched=True,new_native_meta_runs=0,
        qualification='Current complete77-file model+onepreset fullSHA/headers anddeclaredpresetSHA match. No weights leave source. No historical/official provenance,loading,fusion,numerics,training,recovery,performance or effects qualification.')
except BaseException as error:
    import traceback
    dump('failure-private.json',dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:]))
    dump('hash-failure.json',dict(phase=phase,error_type=type(error).__name__,last_progress=events[-1] if events else None,planned_model_files=77,planned_preset_files=1))
    result=dict(passed=False,phase=phase,error_type=type(error).__name__,owner=owner(),epoch=epoch,deadline=deadline,
        wall_seconds=time.monotonic()-start,GPU_cards=0,GPU_GO=False,Model_GO=False)
dump('CPU-result.json',result)
print(json.dumps(dict(passed=result['passed'],phase=phase,error_type=result.get('error_type'),GPU_cards=0)))
if not result['passed']:raise SystemExit(1)
