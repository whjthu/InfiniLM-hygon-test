"""One future metadata-only independent read; no producer imports or artifact IO."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import time

S={'schema': 'original64_metadata_independent_reader_v1', 'source_only': True, 'reader_invocations': 0, 'maximum_invocations': 1, 'request_id': '0fbb14c45b884fbea0950232e850af0c', 'publication': '3973f95839fa84278b756467995c4eb9956e8cc7', 'epoch': 1794907267.640963, 'deadline': 1794908167.640963, 'source_manifest': {'bytes': 1589, 'SHA256': '326922b2f41a7215ab39340bdae3e134983df0b0dfd840edc504b6a83b054378'}, 'source_files': {'canonical/first_update_audit.py': {'bytes': 6939, 'SHA256': '816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49'}, 'check_package_local.py': {'bytes': 4229, 'SHA256': 'c89338319e785bafd6cd10ff9ebf4962be4b90aff8a975ab54280f52a332c30b'}, 'CLAIM.json': {'bytes': 1345, 'SHA256': '9a58f9f26cfc57f35cb3d97975aa19f78be3cddcaac701f39884b7cb80817d7b'}, 'common.py': {'bytes': 2861, 'SHA256': '13f5f84d8478d58170a7131f04a02e3573a1a92ea892a3e184a6bd0c72aeb00c'}, 'derive_inputs.py': {'bytes': 7981, 'SHA256': '13ed6ad6997ffb0620958f32de72090a56a2056d9e4e69f7e5f4bb4032970b41'}, 'launch_once.py': {'bytes': 6600, 'SHA256': '90971f44af23455ef6c6defa5cb7852c904f7a3298625e4211c66ed666336985'}, 'PLAN.json': {'bytes': 4614, 'SHA256': 'daa15b018d9ec4dcfaf3c27455dc8b637808e87bc7c5f41502b9955ff2a653c7'}, 'README.md': {'bytes': 5055, 'SHA256': 'dd3142357d6ab605e91322c196cbb51bfb92bd6698de624c630bccbd255d8849'}, 'receive_once.py': {'bytes': 3415, 'SHA256': '25f25f4aa7f92f0811d30a13cad786c23d0adc34ff879b625806d4bdeba84760'}, 'run_once.py': {'bytes': 7278, 'SHA256': 'ac59c885aa06e53ce96faa3e80a1851097dce49f6b8257382370c67e92deea9c'}, 'SOURCE_PROOF.json': {'bytes': 3736, 'SHA256': 'b8e16df65376f9130defae1426b64303ec34b5064857e32fd6a414aac3c880dc'}}, 'producer_plan': {'label': 'hygon-internvl-A100-input-fingerprints-v1', 'case': 'T03.sft', 'source_install': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-input-fingerprints-v1/source', 'output_namespace': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-input-fingerprints-v1/output', 'namespace': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-input-fingerprints-v1', 'reference_root': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1', 'reference_request_id': '031e04f24efa429595364e6f0da1e5b6', 'original_binding_SHA256': '2ff3b19fc433f38b2b9053787f1b458e70ada2457e1d1092f5beab3ae769a076', 'original_value_audit_SHA256': '9664da5de388ea4687a0e58493e75c72baed61ef96efc90444b2754b7ccdc060', 'fingerprint_source_SHA256': '816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49', 'fingerprint_source_bytes': 6939, 'source_receipts': {'0': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-00/receipt.json', 'bytes': 4302, 'SHA256': '9922d23a0a29037df2e25a3059cbae1082c0021918c248d477c62c3498660c90'}, '1': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-01/receipt.json', 'bytes': 4299, 'SHA256': '9baf5444fc9a31b9cf8546742db00cda75631c94bd1190605af27eb8425165ff'}, '2': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-02/receipt.json', 'bytes': 4301, 'SHA256': '518da3273f76884048eca6efa958c05441b3c67fe26ef1dd8a1b9ce88ee8b22e'}, '3': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-03/receipt.json', 'bytes': 4303, 'SHA256': '741ee896d30d0ac41b6f400372b7a34b7fd9454dd1904b27f098286d8df2fafc'}, '4': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-04/receipt.json', 'bytes': 4299, 'SHA256': '644f684227f9c8450b5d2bcbe635e54a8fb493412e8111397da5a59688a1a291'}, '5': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-05/receipt.json', 'bytes': 4300, 'SHA256': '037dad4b27f3231aaeedfee235fcc44c9c8ff836ead65b24ef37a23bc007cf44'}, '6': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-06/receipt.json', 'bytes': 4300, 'SHA256': 'e313c525860ffd92a6407b2d27877b61acd304719325aed1a99335219a47eb67'}, '7': {'path': '/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-first-updates-v1/capture/rank-07/receipt.json', 'bytes': 4302, 'SHA256': '3586e344b9d2d3a9f6034d3e449176a7b465d01ec160d2465c97f8dc66671959'}}, 'original_capture_bytes_per_rank': [2874894141, 2813645565, 2847236861, 2854853757, 2833408189, 2864032509, 2836375293, 2865993021], 'file_denominator': 64, 'rank_denominator': 8, 'micro_per_rank': 8, 'accumulation': 4, 'per_input_file_limit_bytes': 134217728, 'all_input_file_limit_bytes': 6442450944, 'reference_limit_bytes': 131072, 'metadata_limit_bytes': 131072, 'whole_seconds': 900, 'work_offset_seconds': 780, 'reserve_for_cleanup_seconds': 30, 'CPU_affinity_cores': 2, 'torch_threads': 2, 'torch_interop_threads': 1, 'max_active_workers': 1, 'worker_RSS_sampled_limit_bytes': 8589934592, 'RSS_poll_seconds': 0.2, 'RSS_is_not_hard_limit': True, 'hard_memory_cgroup_enforced': False, 'RLIMIT_AS_enforced': False, 'raw_torch_version': '2.8.0a0+5228986c39.nv25.06', 'distribution_torch_version': '2.8.0a0+5228986c39.nv25.6', 'torch_version_source_SHA256': 'f705f2955ebf304eab692c52e648caa82cfb45ae849979f74bc7ffde19706bd2', 'python': '/home/whj/nvidia-baseline/env-vlm/bin/python', 'fresh_empty_compute_apps_required': True, 'actual_inputs_fingerprints': 'NOT_YET_DERIVED', 'GPU_GO': False, 'target_GO': False, 'CPU_GO': True, 'CPU_GO_condition': 'One future root-reviewed published invocation; CLI publication/MF/claim pins verified against git show and own clean source; fixed exclusive local-only namespace', 'replay_allowed': False, 'remote_bootstrap': '/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -', 'local_runtime_namespace': 'local-only/training-internvl-A100-input-fingerprints-CPU-v1', 'device_compute_calls_planned': 0, 'driver_query_calls_planned': 1}, 'original_outcome': {'bytes': 1350, 'SHA256': '700ab53cc168311de065542c2d23402ee5bc14308b881cfd7b75668213c4e17d'}, 'read_limit_bytes': 1048576, 'reader_seconds': 60, 'known_observed_peak_RSS_bytes': 573763584, 'math_digest_recomputation': False, 'target_GO': False, 'GPU_GO': False, 'remote_calls_in_preparation': 0}
P=S['producer_plan'];ROOT=Path(P['namespace']);READ=0;EVIDENCE={}


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def stamp(st):return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]


def unique(raw):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs)


def take(relative,limit=131072,pin=None):
    global READ
    path=ROOT/relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'relative path')
    require(not any(p.is_symlink() for p in (path,*path.parents)),'linked metadata/source')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        before=os.fstat(f.fileno());require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=limit,'metadata bound')
        require(READ+before.st_size<=S['read_limit_bytes'],'total read bound')
        raw=f.read(limit+1);READ+=len(raw)
        require(len(raw)==before.st_size and stamp(os.fstat(f.fileno()))==stamp(before)==stamp(path.lstat()),'stable metadata FD')
    identity=dict(bytes=len(raw),SHA256=sha(raw));EVIDENCE[relative]=identity
    if pin:require(identity==pin,'original pin mismatch')
    return raw


def obj(relative,limit=131072,pin=None):return unique(take(relative,limit,pin))
def digest(v):return type(v) is str and re.fullmatch('[0-9a-f]{64}',v) is not None


def context(v):
    for key in ('publication','epoch','deadline','source_manifest_SHA256'):
        expected=S['source_manifest']['SHA256'] if key=='source_manifest_SHA256' else S[key]
        require(v[key]==expected,'original context '+key)
    require(v.get('request_id')==S['request_id'],'original request')
    require(v['deadline']-v['epoch']==900,'original epoch900')


def owner_shape(v):
    require(type(v) is dict and set(v)=={'PID','PPID','PGID','startticks','UID'},'owner schema')
    require(all(type(x) is int and x>=0 for x in v.values()) and v['PID']>0 and v['startticks']>0,'owner identity')


def original_owner_absent(v):
    owner_shape(v);path=Path('/proc')/str(v['PID'])/'stat'
    try:
        with path.open('rb') as stream:raw=stream.read(4097)
    except FileNotFoundError:return dict(PID=v['PID'],original_startticks=v['startticks'],original_absent=True,state='PID_ABSENT')
    require(len(raw)<=4096,'known PID stat bound')
    parts=raw.decode().rsplit(') ',1)[1].split();ticks=int(parts[19])
    require(ticks!=v['startticks'],'original owner still exists')
    return dict(PID=v['PID'],original_startticks=v['startticks'],observed_startticks=ticks,original_absent=True,state='PID_REUSED_ORIGINAL_ABSENT')


def main():
    started=time.time()
    def expired(*unused):raise TimeoutError('reader60s')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,S['reader_seconds'])
    source=obj('source/SOURCE_FILES.json',131072,S['source_manifest'])
    require(source['files']==S['source_files'],'source inventory')
    for name,pin in S['source_files'].items():take('source/'+name,256*1024,pin)
    install=obj('install-intent.json');intent=obj('output/intent.json')
    outcome=obj('output/outcome.json',16384,S['original_outcome'])
    worker_owner=obj('output/worker-owner.json',4096);worker=obj('output/worker-result.json',16384)
    resolved=obj('output/resolved-input-pins.json');doc=obj('output/input-fingerprints.json',131072,worker['output'])
    for v in (install,intent,outcome):context(v)
    require(install['owner']==intent['owner'],'supervisor original identity')
    require(install['source_files']==len(S['source_files']) and install['source_bytes']==sum(v['bytes'] for v in S['source_files'].values())+S['source_manifest']['bytes'],'installed source denominator')
    require(intent['work_deadline']==S['epoch']+780 and intent['files']==64 and intent['max_workers']==1,'work scope')
    require(len(intent['CPU_affinity'])==2 and len(set(intent['CPU_affinity']))==2,'CPU scope')
    require(intent['worker_RSS_sampled_limit_bytes']==8*1024**3 and intent['hard_memory_cgroup_enforced'] is False and intent['RLIMIT_AS_enforced'] is False,'memory terminology')
    require(outcome['passed'] is True and outcome['worker_returncode']==0 and outcome['own_worker_reaped'] is True and outcome['failure_type'] is None and outcome['cleanup_error_type'] is None,'terminal success')
    require(outcome['completed_files']==outcome['file_denominator']==64 and outcome['worker_result']==worker and outcome['worker']==worker_owner,'worker chain')
    owner_shape(worker_owner);owner_shape(intent['owner'])
    require(worker_owner['PPID']==intent['owner']['PID'] and worker_owner['PGID']==worker_owner['PID'],'owned worker parent/session')
    require(outcome['observed_worker_peak_RSS_bytes']==S['known_observed_peak_RSS_bytes'] and outcome['RSS_sampling_seconds']==.2,'original observed RSS')
    require(outcome['device_compute_calls']==0 and outcome['driver_query_calls']==1 and outcome['target_GO'] is False and outcome['other136_payloads_read']==0 and outcome['RNG_equivalence'] is False,'producer boundaries')
    require(worker['passed'] is True and worker['completed_files']==worker['file_denominator']==64 and worker['input_payload_only'] is True and worker['other136_payloads_read']==0 and worker['models']==0 and worker['device_compute_calls']==worker['driver_query_calls']==0,'worker success boundaries')
    required={'schema','case','reference_request_id','original_binding_SHA256','original_value_audit_SHA256','fingerprint_source_SHA256','projection','world_size','accumulation','source_receipts','rows'}
    require(type(doc) is dict and set(doc)==required,'document schema')
    require(doc['schema']=='native_first_two_inputs_v1' and doc['case']=='T03.sft' and doc['projection']==['args','kwargs'] and doc['world_size']==8 and doc['accumulation']==4,'input protocol')
    for key in ('reference_request_id','original_binding_SHA256','original_value_audit_SHA256','fingerprint_source_SHA256','source_receipts'):require(doc[key]==P[key],'document provenance '+key)
    require(resolved['case']=='T03.sft' and resolved['source_receipts']==P['source_receipts'] and resolved['all_pins_resolved_before_Torch'] is True,'resolved receipt provenance')
    require(set(resolved['receipt_stamps'])==set(str(n) for n in range(8)),'eight receipt stamps')
    for rank,identity in resolved['receipt_stamps'].items():
        require(type(identity) is list and len(identity)==5 and all(type(x) is int for x in identity) and identity[2]==P['source_receipts'][rank]['bytes'],'typed original receipt stamps')
    require(type(doc['rows']) is list and len(doc['rows'])==64 and type(resolved['rows']) is list and len(resolved['rows'])==64,'full64 denominator')
    total=0
    for index,(row,original) in enumerate(zip(doc['rows'],resolved['rows'])):
        rank,micro=divmod(index,8);micro+=1
        require(type(row) is dict and set(row)=={'rank','micro_ordinal','global_step','path','bytes','file_SHA256','input_fingerprint_SHA256'},'row schema')
        require(all(type(row[k]) is int for k in ('rank','micro_ordinal','global_step','bytes')),'typed row integers')
        require((row['rank'],row['micro_ordinal'],row['global_step'])==(rank,micro,(micro-1)//4),'rank/micro/step ordered64')
        require(row['path']==P['reference_root']+'/capture/rank-%02d/forward-%02d-input.pt'%(rank,micro),'exact original input path')
        require(0<row['bytes']<=P['per_input_file_limit_bytes'] and digest(row['file_SHA256']) and digest(row['input_fingerprint_SHA256']),'typed SHA/file bounds')
        require(type(original) is dict and set(original)==(set(row)-{'input_fingerprint_SHA256'})|{'identity'},'resolved row schema')
        require({k:v for k,v in row.items() if k!='input_fingerprint_SHA256'}=={k:v for k,v in original.items() if k!='identity'},'fullfile pin cross-resolved')
        require(type(original['identity']) is list and len(original['identity'])==5 and all(type(x) is int for x in original['identity']) and original['identity'][2]==row['bytes'],'resolved stable identity')
        total+=row['bytes']
    require(total==resolved['total_input_bytes']==worker['input_bytes'] and total<=P['all_input_file_limit_bytes'],'total bytes')
    posts=[original_owner_absent(intent['owner']),original_owner_absent(worker_owner)]
    result=dict(passed=True,request_id=S['request_id'],publication=S['publication'],original_context=dict(epoch=S['epoch'],deadline=S['deadline']),
        original_outcome=S['original_outcome'],source_manifest=S['source_manifest'],metadata_file_pins=EVIDENCE,metadata_bytes_read=READ,
        rank_denominator=8,file_denominator=64,input_bytes=total,input_reference_pin=worker['output'],input_reference=doc,
        known_owner_posts=posts,observed_producer_peak_RSS_bytes=S['known_observed_peak_RSS_bytes'],reader_elapsed_seconds=time.time()-started,
        independent_metadata_audit=True,producer_canonical_source_qualified=True,independent_math_digest_recomputed=False,
        target_must_recompute_actual_inputs=True,target_GO=False,GPU_GO=False,reader_Torch_imports=0,reader_device_compute_calls=0,reader_driver_query_calls=0,
        original64_payloads_read=0,other136_payloads_read=0,replay_allowed=False)
    raw=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode();require(len(raw)<=131072,'reader output cap')
    print(raw.decode(),flush=True);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,metadata_file_pins=EVIDENCE,metadata_bytes_read=READ,request_id=S['request_id'],replay_allowed=False,target_GO=False)),flush=True)
        raise SystemExit(1)
