"""Prepare control45 only on A100 from exact existing generic64, never export captions."""
import hashlib
import json
from pathlib import Path,PurePosixPath
from identities import select_generic45

D=Path(__file__).resolve().parent


def need(ok,why):
    if not ok:raise ValueError(why)

def canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def pinned(path,pin,cap=2*1024**2):
    need(path.is_file() and not path.is_symlink() and 0<pin['bytes']<=cap,'Bounded regular source required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
        and len(raw)==pin['bytes'] and digest(raw)==pin['SHA256'],'Source identity changed')
    return raw

def verify_source():
    manifest=json.loads((D/'SOURCE_FILES.json').read_bytes())
    for name,pin in manifest['files'].items():
        need(Path(name).name==name and name not in {'.','..'},'Source path differs');pinned(D/name,pin)

def transform_selected(allrows,selected,target):
    need(len(allrows)==64 and len(selected)==45,'Original64/control45 full denominator')
    need(len({r['original_ordinal'] for r in selected})==45 and len({r['image_SHA256'] for r in selected})==45,'Repeated selected identity')
    rows=[];provenance=[]
    for r in selected:
        original=allrows[r['original_ordinal']];need(original['images']==[r['source']],'Original ordinal/image identity changed')
        messages=original['messages'];need(len(messages)==2 and messages[0]['role']=='user' and messages[1]['role']=='assistant'
            and isinstance(messages[0]['content'],str) and isinstance(messages[1]['content'],str) and messages[1]['content'].strip(),'Full original messages schema differs; no guessing')
        original_answer=messages[1]['content'];answer=original_answer if original_answer.startswith('FINAL: ') else 'FINAL: '+original_answer
        name='train/media/'+PurePosixPath(r['source']).name
        identifier='generic-control45:'+str(r['original_ordinal']).zfill(2)+':'+r['image_SHA256']
        rows.append(dict(id=identifier,messages=[dict(messages[0]),dict(messages[1],content=answer)],images=[str(PurePosixPath(target)/name)]))
        provenance.append(dict(source_ordinal=r['original_ordinal'],source_id_SHA256=digest(identifier.encode()),image_name=name,image_SHA256=r['image_SHA256'],
            original_answer_SHA256=digest(original_answer.encode()),transformed_answer_SHA256=digest(answer.encode()),FINAL_prefix_added=answer!=original_answer))
    return rows,provenance

def prepare_control(scope,metadata_raw,plan,check):
    need(scope.is_absolute() and scope.resolve()==scope,'Owned absolute new namespace required')
    mp=plan['generic64_metadata'];need(len(metadata_raw)==mp['bytes'] and digest(metadata_raw)==mp['SHA256'],'Frozen original64 metadata differs')
    # Fixed metadata-only selection completes before original64 captions are opened.
    roster=select_generic45(json.loads(metadata_raw),plan['selection_seed']);need(digest(canonical(roster))==plan['roster_SHA256'],'Frozen selected45 roster changed')
    # Reuse the ten already qualified dev records; no dev image/encode/gold replay.
    refs=plan['shared_dev_reference'];check()
    shared_native=json.loads(pinned(Path(refs['native_identities']['path']),refs['native_identities'],128*1024))
    dev_reference=[r for r in shared_native if r['split']=='dev']
    need(len(shared_native)==55 and len(dev_reference)==10 and all(r['labels_None'] is True and r['full_input_retained'] is True for r in dev_reference),'Saved shared dev10 native identity differs')
    oldbinding=json.loads(pinned(Path(refs['data_binding']['path']),refs['data_binding'],8192))
    need(oldbinding['passed'] is True and oldbinding['files']['dev-swift.jsonl']['SHA256']==refs['dev_input']['SHA256'] and oldbinding['dev_gold_read'] is False and oldbinding['blind_inputs_gold_or_media_read'] is False,'Saved shared dev binding differs')
    pinned(Path(refs['dev_input']['path']),refs['dev_input'],8192)
    check();original=pinned(Path(plan['generic_original_dataset']['path']),plan['generic_original_dataset'],32768)
    allrows=[json.loads(line) for line in original.splitlines()];need(len(allrows)==64,'Original generic64 row denominator differs')
    target=scope/'data';need(not target.exists(),'Control data namespace exists');target.mkdir(mode=0o700)
    rows,provenance=transform_selected(allrows,roster['selected'],target.as_posix());media={};total=0
    def progress(phase):
        value=dict(phase=phase,source_rows=64,selected_rows=45,planned_native_rows=45,completed_native_rows=0,
            planned_images=45,completed_images=len(media),actual_image_bytes=total,shared_dev_rows_reused=10,shared_dev_new_encodes=0,
            caption_payload_exported=False,dev_gold_read=False,blind_data_read=False)
        temp=target/'prepare-progress.tmp';temp.write_bytes(canonical(value));temp.replace(target/'prepare-progress.json')
    progress('SELECTED45_SOURCE64_PINNED')
    for r in roster['selected']:
        check();raw=pinned(Path(r['source']),dict(bytes=r['image_bytes'],SHA256=r['image_SHA256']),1024**2)
        need(raw.startswith(b'\xff\xd8') and raw.endswith(b'\xff\xd9'),'Original JPEG framing changed')
        name='train/media/'+Path(r['source']).name;dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
        media[name]=dict(name=name,bytes=len(raw),SHA256=r['image_SHA256']);total+=len(raw);progress('SELECTED_MEDIA_COPY')
    need(len(media)==45 and total==plan['control_image_bytes']==7321484,'Full selected45 media bytes differ')
    raw=b''.join(canonical(r) for r in rows)
    with (target/'train-swift.jsonl').open('xb') as f:f.write(raw)
    ledger=canonical(dict(original_JSONL_SHA256=digest(original),roster_SHA256=plan['roster_SHA256'],selected=provenance,source_count=64,selected_count=45,caption_payload_exported_to_PC=False,
        original_question_and_caption_full=True,response_prefix='FINAL: ',actual_GPU=0,blind_or_dev_gold_read=False))
    with (target/'control-provenance-private.json').open('xb') as f:f.write(ledger)
    binding=dict(passed=True,state='GENERIC_CONTROL45_FINAL_ADAPTER_ONLY',target_data=target.as_posix(),generic_original_dataset_SHA256=digest(original),generic_roster_SHA256=plan['roster_SHA256'],
        domain_native_qualification_SHA256=plan['native_declared_qualification']['SHA256'],shared_dev_rows_reused=10,shared_dev_new_encodes=0,
        shared_dev_native_identities_SHA256=refs['native_identities']['SHA256'],shared_dev_input_SHA256=refs['dev_input']['SHA256'],shared_dev_rows_SHA256=digest(canonical(dev_reference)),
        files={'train-swift.jsonl':dict(bytes=len(raw),SHA256=digest(raw),rows=45)},media=media,provenance_SHA256=digest(ledger),
        dev_gold_read=False,blind_inputs_gold_or_media_read=False,original_format_modified=True,FINAL_prefix_added_to_original_caption=True,
        caption_payload_exported_to_PC=False,literal_special_tokens_added=False,native_qualified=False,training_GO=False,GPU_GO=False)
    db=canonical(binding)
    with (target/'data-binding-private.json').open('xb') as f:f.write(db)
    progress('CONTROL45_STAGED_NATIVE_PENDING')
    return db
