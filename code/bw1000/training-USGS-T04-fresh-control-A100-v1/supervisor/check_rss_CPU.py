"""Only RSS-v2 new boundaries; pure dictionaries and bytes, no proc or subprocess."""
import copy
import bounded_GPU_supervisor as s


def run():
    checks=[]
    owner=dict(PID=44,startticks=987)
    live=dict(owner,state='S');exit=dict(owner,state='Z')
    def measure(raw,before=live,after=live):
        return s.checked_owned_rss(owner,before,lambda:raw,lambda:after)
    def state():return dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_exceeded=False)
    def sample(resources,row):return s.sample_group_rss({44:owner},resources,reader=lambda _:row)
    good=measure(b'State:\tS (sleeping)\nVmRSS:\t42 kB\n')
    assert good['classification']=='MEASURED' and good['RSS_bytes']==42*1024
    r=state();assert sample(r,good) and r['last_sample']['coverage_complete'];checks.append('normal_exact_rss')
    for raw,after,label in [(b'State:\tZ (zombie)\n',live,'status_exit'),(b'State:\tS (sleeping)\n',exit,'after_exit'),(b'',exit,'after_exit_status_already_empty')]:
        row=measure(raw,after=after);assert row['classification']=='EXITED_DURING_SAMPLE' and row['RSS_bytes'] is None
        r=state();assert sample(r,row) and r['last_sample']['matched_processes']==0 and r['last_sample']['unknown_live_processes']==0;checks.append(label)
    def forbidden():raise AssertionError('status of exited/reused process must not be read')
    row=s.checked_owned_rss(owner,exit,forbidden,forbidden);assert row['classification']=='EXITED_BEFORE';checks.append('before_exit_no_status')
    row=s.checked_owned_rss(owner,dict(live,startticks=988),forbidden,forbidden)
    assert row['classification']=='IDENTITY_CHANGED_BEFORE';checks.append('identity_reuse_no_status')
    row=measure(b'State:\tS (sleeping)\n',after=dict(live,startticks=988));assert row['classification']=='IDENTITY_CHANGED_AFTER';checks.append('identity_reuse_after')
    missing=measure(b'State:\tS (sleeping)\n')
    r=state();assert sample(r,missing) and r['last_sample']['group_RSS_bytes'] is None and r['last_sample']['known_group_RSS_bytes']==0 and not r['last_sample']['coverage_complete'];checks.append('first_missing_unknown_not_zero')
    try:sample(r,copy.deepcopy(missing))
    except ValueError:pass
    else:raise AssertionError('persistent live missing accepted')
    assert r['samples']==2 and r['RSS_last_failure']['classification']=='LIVE_MISSING_RSS_PERSISTENT' and r['RSS_last_failure']['status']['VmRSS_row_count']==0;checks.append('second_poll_missing_fail_recorded')
    r=state();sample(r,copy.deepcopy(missing));sample(r,good)
    assert not r['RSS_missing_pending'] and r['last_sample']['coverage_complete'];checks.append('one_followup_measured_recovery')
    try:sample(r,copy.deepcopy(missing))
    except ValueError:pass
    else:raise AssertionError('same owner second missing allowance accepted')
    assert r['RSS_last_failure']['classification']=='LIVE_MISSING_RSS_ALLOWANCE_USED';checks.append('sameowner_allowance_never_reset')
    for raw,label in [(b'State:\tS\nVmRSS:\t1 kB\nVmRSS:\t2 kB\n','duplicate'),(b'State:\tS\nVmRSS:\t2 MB\n','unit'),(b'State:\tS\nVmRSS:\t-2 kB\n','numeric'),(b'VmRSS:\t2 kB\n','missing_state'),(b'State:\tZ\nVmRSS:\tbad kB\n','exit_present_malformed')]:
        row=measure(raw);r=state()
        try:sample(r,row)
        except ValueError:pass
        else:raise AssertionError('malformed accepted '+label)
        assert r['RSS_last_failure']['classification'].startswith('MALFORMED_');checks.append(label+'_failclosed')
    row=copy.deepcopy(good);row['RSS_bytes']=s.RSS_SOFT_BYTES+1;r=state()
    assert not sample(r,row) and r['soft_limit_exceeded'];checks.append('256GiB_unchanged_soft_stop')
    assert all('raw' not in x for x in r.get('RSS_diagnostics',[]))
    return dict(passed=True,count=len(checks),checks=checks,old_tests_repeated=False,actual_proc_reads=0,child_spawns=0,Torch_imports=0,remote=0,GPU=0)


if __name__=='__main__':
    import json
    print(json.dumps(run(),sort_keys=True))
