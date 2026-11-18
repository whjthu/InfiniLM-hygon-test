"""Pre-registered independent small-metadata audit. Never reads an asset body."""
import argparse,hashlib,json,math,os,re,signal,stat,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent

def need(v,why):
    if not v:raise ValueError(why)

def main(args):
    begin=time.monotonic();need(math.isfinite(args.read_epoch) and 0<=time.time()-args.read_epoch<45,'READ_ORIGIN')
    end=begin+args.read_epoch+45-time.time();signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,args.read_epoch+43-time.time()))
    pins={};total=0;errors=[];files={};source={};row=None;qualified=False
    def left():
        v=min(end-time.monotonic(),args.read_epoch+43-time.time());need(v>0,'READ_DEADLINE');return v
    def take(name,path,cap=262144):
        nonlocal total
        left();need(not path.is_symlink(),'METADATA_LINK');a=path.lstat();need(stat.S_ISREG(a.st_mode) and a.st_size<=cap and total+a.st_size<=2097152,'METADATA_BOUND')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        def key(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid)
        try:
            need(key(a)==key(os.fstat(fd)),'METADATA_FD');parts=[];n=0
            while n<a.st_size:
                left();b=os.read(fd,min(65536,a.st_size-n));need(b,'METADATA_SHORT');n+=len(b);total+=len(b);parts.append(b)
            need(not os.read(fd,1) and key(a)==key(os.fstat(fd))==key(path.lstat()),'METADATA_STABLE')
        finally:os.close(fd)
        b=b''.join(parts);pins[name]=dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest());return b
    try:
        cpus=sorted(os.sched_getaffinity(0)-{0,1});need(len(cpus)>=2,'READ_TWO_CPU');os.sched_setaffinity(0,cpus[:2])
        mfraw=take('source/SOURCE_FILES.json',D/'SOURCE_FILES.json');need(hashlib.sha256(mfraw).hexdigest()==args.source_manifest_sha256,'EXPECTED_PUBLISHED_MF')
        mf=json.loads(mfraw)['files']
        for name,pin in mf.items():
            b=take('source/'+name,D/name);source[name]=b;need(pins['source/'+name]==pin,'PUBLISHED_SOURCE_PIN')
        plan=json.loads(source['PLAN.json']);run=Path(plan['RUN'])
        ctxkeys={'request_id':args.request,'publication':args.publication,'epoch':args.epoch,'deadline':args.epoch+300,'source_manifest_SHA256':args.source_manifest_sha256}
        for name in ('install-intent.json','control-state.json','step-intent.json','outcome.json','audit/intent.json','audit/inventory.json','audit/outcome.json'):
            path=run/name
            if not path.exists():files[name]=None;continue
            q=json.loads(take(name,path));files[name]=q;need(all(q[k]==v for k,v in ctxkeys.items()),'ORIGINAL_CONTEXT_'+name)
        s=files.get('step-intent.json');o=files.get('outcome.json');a=files.get('audit/outcome.json');inv=files.get('audit/inventory.json')
        need(s and o and a and inv,'MISSING_ORIGINAL_METADATA')
        need(files['control-state.json']['exec_request_attempted'] is True,'ORIGINAL_ATTEMPT')
        need(s['Slurm_seconds'] in (120,180,240) and s['step_end']<=args.epoch+270 and s['work_deadline']<=s['step_end']-10,'SHRUNK_SLURM_WORK_BUDGET')
        cols='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'
        q=subprocess.run(['sacct','-n','-P','-j',plan['parent'],'-o',cols],capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=min(5,left()))
        need(q.returncode==0 and len(q.stdout)<=1048576,'BOUNDED_TERMINAL_ACCOUNTING')
        rows=[v for line in q.stdout.splitlines() if len(v:=line.split('|'))==7 and v[1]==plan['label']];need(len(rows)==1,'ONE_EXACT_LABEL');row=rows[0]
        need(re.fullmatch(plan['parent']+r'\.\d+',row[0]) and row[2:4]==['COMPLETED','0:0'] and row[5]==plan['node'] and set(row[6].split(','))=={'cpu=4','mem=4G','node=1'},'ACTUAL_TERMINAL_RESOURCE')
        need(o['accounting']==row and a['step']==row[0] and o['held_srun']['reaped'] is True and o['held_srun']['termination_needed'] is False and o['held_srun']['returncode']==0,'ORIGINAL_ACCOUNTING_HELD')
        need(o['node']==a and o['failure'] is None and a['error'] is None and a['passed'] is True,'PRODUCER_CONSISTENCY')
        need(a['planned_files']==26 and a['planned_hash_files']==a['attempted_hash_files']==a['completed_hash_files']==14 and a['missing_hash_files']==0 and a['actual_hash_bytes']==plan['hash_bytes']==4922323586,'FULL14_DENOMINATOR')
        expected={v['name']:v for v in plan['files']};observed={v['name']:v for v in inv['final_files']};members=a['verified_members'];need(len(observed)==26 and set(observed)==set(expected) and len(members)==14 and [v['name'] for v in members]==plan['hash_names'],'EXACT26_14')
        for v in members:
            p=expected[v['name']];t=v['current_target_stat'];i=observed[v['name']]
            need(v['SHA256']==p['SHA256'] and v['bytes']==p['bytes']==t['st_size'] and i['stat']==t and i['exists'] is True and v['whole_file_read_once'] is True,'CURRENT14_HASH_FD_EVIDENCE')
            need(v['observer_host']==a['owner']['host']==plan['node'] and v['observer_boot_id']==a['owner']['boot_id'] and a['owner']['UID']==154459 and stat.S_ISREG(t['st_mode']) and t['st_uid']==154459,'CURRENT_OBSERVER_IDENTITY')
        need(inv['partial']['name']==plan['partial_name'] and inv['cross_host_saved_st_dev_not_compared'] is True,'PARTIAL_STAT_SCOPE')
        r=a['resources'];need(len(r['CPU_affinity'])==2 and not ({0,1}&set(r['CPU_affinity'])) and r['AS_is_not_RSS'] is True,'WORKER_RESOURCE')
        need(all(a[k]==0 for k in ('old_payload_writes','hardlinks_created','transfer_bytes','A100_calls','Torch_SDK_GPU_calls')) and a['all26_assets_qualified'] is False,'READONLY_RESULT_SCOPE')
        qualified=True
    except BaseException as e:errors.append(dict(type=type(e).__name__,reason_SHA256=hashlib.sha256(str(e).encode()).hexdigest()))
    result=dict(request_id=args.request,publication=args.publication,epoch=args.epoch,source_manifest_SHA256=args.source_manifest_sha256,independent_metadata_verified=qualified,planned_hash_files=14,observed_hash_files=(files.get('audit/outcome.json') or {}).get('completed_hash_files'),pins=pins,metadata_read_bytes=total,accounting=row,errors=errors,asset_body_reads=0,independent_body_rehash=False,node_process_absence='NOT_OBSERVED_ACCOUNTING_TERMINAL_ONLY',all26_assets_qualified=False,GPU_GO=False,Model_GO=False,replay_allowed=False,elapsed_monotonic_seconds=time.monotonic()-begin)
    print(json.dumps(result,sort_keys=True,allow_nan=False));signal.setitimer(signal.ITIMER_REAL,0);return 0 if qualified else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--publication',required=True);p.add_argument('--epoch',required=True,type=float);p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--read-epoch',required=True,type=float);raise SystemExit(main(p.parse_args()))
