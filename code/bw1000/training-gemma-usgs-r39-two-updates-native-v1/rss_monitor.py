"""Exact audited RSS-v2 functions, Linux/HIP-independent; sampled soft limit only."""
import os,hashlib,time
RSS_SOFT_BYTES = 256 * 1024**3

def need(ok, code):
    if not ok:
        raise ValueError(code)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def rss_record(saved, before=None, status=None, after=None, *, classification, value=None):
    def phase(row):
        if row is None:
            return None
        return {key:row.get(key) for key in ('PID','startticks','state')}
    return dict(PID=saved['PID'],startticks=saved['startticks'],before=phase(before),
                status=status,after=phase(after),classification=classification,RSS_bytes=value)

def checked_owned_rss(saved, before, status_reader, after_reader):
    # No status read for a PID that no longer belongs to this registry identity.
    match=lambda row: row['PID']==saved['PID'] and row['startticks']==saved['startticks']
    if not match(before):
        return rss_record(saved,before,classification='IDENTITY_CHANGED_BEFORE')
    if before['state'] in ('Z','X'):
        return rss_record(saved,before,classification='EXITED_BEFORE')
    raw=status_reader()
    lines=raw.splitlines()
    states=[line for line in lines if line.startswith(b'State:')]
    rows=[line for line in lines if line.startswith(b'VmRSS:')]
    status_state=None
    if len(states)==1:
        parts=states[0].split()
        if len(parts)>=2 and len(parts[1])==1 and parts[1] in (b'R',b'S',b'D',b'Z',b'T',b't',b'X',b'I',b'P',b'K',b'W'):
            status_state=parts[1].decode('ascii')
    status=dict(state=status_state,State_row_count=len(states),VmRSS_row_count=len(rows),
                bytes=len(raw),SHA256=sha(raw),State_rows_SHA256=sha(b'\n'.join(states)),VmRSS_rows_SHA256=sha(b'\n'.join(rows)))
    after=after_reader()
    if not match(after):
        return rss_record(saved,before,status,after,classification='IDENTITY_CHANGED_AFTER')
    # A present malformed metric is never silently accepted, even during exit.
    if len(rows)>1:
        return rss_record(saved,before,status,after,classification='MALFORMED_VMRSS_DUPLICATE')
    value=None
    if rows:
        parts=rows[0].split()
        if len(parts)!=3 or parts[1].isdigit() is False or parts[2]!=b'kB':
            return rss_record(saved,before,status,after,classification='MALFORMED_VMRSS_VALUE_UNIT')
        value=int(parts[1])*1024
    terminal=after['state'] in ('Z','X') or status_state in ('Z','X')
    if terminal:
        # Do not manufacture a 0-byte measurement for an exited process.
        return rss_record(saved,before,status,after,classification='EXITED_DURING_SAMPLE')
    if status_state is None:
        return rss_record(saved,before,status,after,classification='MALFORMED_STATUS_STATE')
    if not rows:
        return rss_record(saved,before,status,after,classification='LIVE_MISSING_RSS')
    return rss_record(saved,before,status,after,classification='MEASURED',value=value)

def read_owned_rss(saved):
    directory=None
    before=None
    try:
        directory=os.open('/proc/'+str(saved['PID']),os.O_RDONLY|os.O_DIRECTORY)
        need(os.fstat(directory).st_uid==os.getuid(),'OWNED_PROC_UID')
        def read(name,cap):
            fd=os.open(name,os.O_RDONLY,dir_fd=directory)
            try:
                with os.fdopen(fd,'rb',closefd=False) as stream:
                    raw=stream.read(cap+1)
                need(len(raw)<=cap,'OWNED_PROC_RECORD_BOUND')
                return raw
            finally:
                os.close(fd)
        def identity():
            raw=read('stat',8192).decode('ascii')
            fields=raw.rsplit(') ',1)[1].split()
            return dict(PID=int(raw.split(' ',1)[0]),startticks=int(fields[19]),state=fields[0])
        before=identity()
        return checked_owned_rss(saved,before,lambda:read('status',65536),identity)
    except (FileNotFoundError,ProcessLookupError):
        return rss_record(saved,before,classification='ABSENT_DURING_SAMPLE')
    finally:
        if directory is not None:
            os.close(directory)

def sample_group_rss(registry,resources,reader=read_owned_rss):
    total=0;sampled=0;unavailable=0;unknown=0;failures=[];counts={}
    pending=resources.setdefault('RSS_missing_pending',{})
    diagnostics=resources.setdefault('RSS_diagnostics',[])
    seen=resources.setdefault('RSS_missing_seen',{})
    for saved in registry.values():
        row=reader(saved)
        need(row['PID']==saved['PID'] and row['startticks']==saved['startticks'],'RSS_READER_OWNER')
        key=str(saved['PID'])+':'+str(saved['startticks'])
        kind=row['classification']
        if kind=='LIVE_MISSING_RSS':
            unknown+=1
            if key in seen:
                kind='LIVE_MISSING_RSS_PERSISTENT' if key in pending else 'LIVE_MISSING_RSS_ALLOWANCE_USED'
                row['classification']=kind
                failures.append(kind)
            else:
                need(len(seen)<1024,'RSS_PENDING_OWNER_BOUND')
                pending[key]=dict(first_sample=resources['samples']+1,confirmations_allowed=1)
                seen[key]=resources['samples']+1
        elif kind=='MEASURED':
            pending.pop(key,None)
            value=row['RSS_bytes'];need(type(value)is int and value>=0,'OWNED_RSS_VALUE')
            total+=value;sampled+=1
        elif kind in ('EXITED_BEFORE','EXITED_DURING_SAMPLE','ABSENT_DURING_SAMPLE','IDENTITY_CHANGED_BEFORE','IDENTITY_CHANGED_AFTER'):
            pending.pop(key,None);unavailable+=1
        else:
            unknown+=1;failures.append(kind)
        counts[kind]=counts.get(kind,0)+1
        if kind!='MEASURED':
            diagnostics.append(dict(sample=resources['samples']+1,**row))
            if len(diagnostics)>128:
                del diagnostics[:-128]
            resources['RSS_diagnostic_events']=resources.get('RSS_diagnostic_events',0)+1
            if kind in failures:
                resources['RSS_last_failure']=dict(sample=resources['samples']+1,**row)
    resources['samples']+=1
    resources['sampled_group_peak_RSS_bytes']=max(resources['sampled_group_peak_RSS_bytes'],total)
    resources['RSS_peak_is_observed_lower_bound']=True
    if unknown:
        resources['RSS_unknown_coverage_samples']=resources.get('RSS_unknown_coverage_samples',0)+1
    resources['last_sample']=dict(group_RSS_bytes=None if unknown else total,known_group_RSS_bytes=total,
        coverage_complete=unknown==0,unknown_live_processes=unknown,matched_processes=sampled,
        exited_absent_or_identity_changed=unavailable,classifications=counts,observed_monotonic=time.monotonic())
    resources['soft_limit_exceeded'] |= total>RSS_SOFT_BYTES
    need(not failures,'OWNED_RSS_SCHEMA_OR_PERSISTENT_MISSING')
    return total<=RSS_SOFT_BYTES
