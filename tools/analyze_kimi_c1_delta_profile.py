"""Read-only terminal analysis of the bounded current C1 Delta component."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,math,statistics,subprocess
LABEL='Kimi-C1-Delta-GPU-r311'
MANIFEST='db1e7416832440ae880cfbe0c05e3c76fb68e1e3c55a9aa479222667e9271af0'
CPU='81cf85498a36a768089bfca4019173e2a862034a7dfee731a954159df4662080'
SHIM='81f950279a83dbb5e581b2952caea9a425822a185f6fae3a5c97894b57388e8c'
MAIN='c6a9bd3f79b32fabc1034974102c68b7fab7db0dcbc3e98ac0474745662df46e'
REMOTE=PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
ARMS=['OFF-before','ON','OFF-after']


def require(ok,message):
    if not ok:raise ValueError(message)


def stats(values):
    require(values and all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in values),'Invalid timing')
    return {'n':len(values),'min':min(values),'median':statistics.median(values),'max':max(values)}


def summarize(workers):
    require(len(workers)==16 and sorted(v['rank']for v in workers)==list(range(16)),'Need unique16owners')
    for v in workers:
        require(v['completed'] and v['normal_close'] and v['layer_calls']==31 and 'error'not in v,'Incomplete owner')
        samples=v['samples'];require(len(samples)==30,'Incomplete sample sequence')
        require([(x['arm'],x['trial'],x['warmup'])for x in samples]==
                [(arm,t,t<2)for arm in ARMS for t in range(10)],'Arm/sample order differs')
        for sample in samples:
            require(sample['output_state_equal_to_OFF'] and sample['source_rows_unchanged'],'Output or state changed')
            require(math.isfinite(sample['API_wall_ms']) and sample['API_wall_ms']>0,'Invalid APIwall')
            p=sample['profile']
            if sample['arm']=='ON':
                require(p['C']==p['T']==1 and p['pool']==3 and p['added_segment_syncs']==0,'Wrong actual profile geometry')
                segments=p['segments'];require(segments and all(math.isfinite(s['stream_elapsed_ms'])and s['stream_elapsed_ms']>=0 for s in segments),'Bad segment')
                require(math.isclose(sum(s['stream_elapsed_ms']for s in segments),p['stream_total_ms'],rel_tol=1e-4,abs_tol=.005),'Adjacent segments do not cover stream')
            else:require(p is None,'OFF arm accidentally profiled')
    maxwall={};critical=[]
    for i,arm in enumerate(ARMS):
        maxwall[arm]=stats([max(v['samples'][i*10+t]['API_wall_ms']for v in workers)for t in range(2,10)])
    for t in range(2,10):
        v=max(workers,key=lambda v:v['samples'][10+t]['profile']['stream_total_ms'])
        p=v['samples'][10+t]['profile']
        critical.append({'trial':t,'rank':v['rank'],'stream_total_ms':p['stream_total_ms'],
                         'segments_ms':{s['name']:s['stream_elapsed_ms']for s in p['segments']}})
    keys=set(critical[0]['segments_ms']);require(all(set(x['segments_ms'])==keys for x in critical),'Phase layout differs')
    phases={k:stats([x['segments_ms'][k]for x in critical])for k in sorted(keys)}
    a,b,c=[maxwall[arm]['median']for arm in ARMS];ref=(a+c)/2;ratio=b/ref;spread=abs(a-c)/ref
    return {'measured_samples_per_arm':8,'warmups_per_arm':2,'max16owner_APIwall_ms':maxwall,
            'APIwall_ON_over_OFF_bracket':ratio,'APIwall_OFF_relative_spread':spread,
            'APIwall_absolute_bracket_within3pct':abs(ratio-1)<=.03 and spread<=.03,
            'ON_critical_owner_per_trial':critical,'ON_phase_ms_at_same_selected_critical_owner':phases,
            'ON_max16owner_stream_total_ms':stats([x['stream_total_ms']for x in critical]),
            'largest_observed_ON_phase':max(phases,key=lambda k:phases[k]['median']),
            'new_kernel_or_Model_gain':False,'pure_kernel_time':False,
            'limits':['Historical teacher512 prefix and one token, not current live32K state or all KDA layers.',
                      'APIwall includes H2D/D2H plus ON finalize/JSON; ON/OFF ratio measures observer cost, not optimization gain.',
                      'Stream intervals include host gaps and TPwait; do not call o_proj+TP pure communication.',
                      'Warmups excluded; eight repeats are descriptive, no significance/generalization claim.',
                      'SingleDelta excludes AttnRes/norm outside this class and cannot replace current wholeModel profile.']}


def analyze(root,accounting,cleanup_recovery=None):
    root=Path(root).resolve();evidence={}
    def read(name,expected=None):
        p=(root/name).resolve();require(p.is_relative_to(root),'Evidence escapes root')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();require(expected is None or h==expected,'Evidence SHA differs')
        evidence[str(name)]=h;return json.loads(raw)
    w=read('results/'+LABEL+'-watch.json')
    require(w['completed'] and w['one_attempt'],'Incomplete terminal')
    require(w['source_manifest_SHA256']==MANIFEST and w['CPU_receipt_SHA256']==CPU,'Unexpected source/CPU')
    actual=w['actual_GPU'];keeper=w['CPU_keeper'];require(w['terminal'][0]==actual and w['terminal'][2:4]==['COMPLETED','0:0'],'GPU not terminal0')
    records={x[0]:x for x in accounting}
    require(actual in records and records[actual][2:4]==['COMPLETED','0:0'],'Independent GPU accounting incomplete')
    repair=None
    if w.get('component_lifecycle_qualified') and w.get('post_empty'):
        require(keeper in records and records[keeper][2:4]==['COMPLETED','0:0'],'Independent keeper accounting incomplete')
        post=w['post'];refs=w['owners']
    else:
        require(cleanup_recovery is not None,'Controller failed; explicit independent cleanup receipt required')
        repair=read(cleanup_recovery)
        require(repair['state']=='DELAYED_INDEPENDENT_CLEANUP_PASS_ORIGINAL_KEEPER_FAILURE_RETAINED' and
                repair['actual_GPU']==actual and repair['all_nodes_clean'] is True and
                repair['original_keeper_relabelled'] is False and repair['GPU_replayed'] is False and
                repair['watch_SHA256']==evidence['results/'+LABEL+'-watch.json'], 'Invalid recovery receipt')
        require('[Errno 98] Address already in use' in w['probe_attempts']['post']['stderr'],
                'Recovery only covers independently proven post-port failure')
        require(records[keeper][2:4]==['FAILED','1:0'] and repair['original_keeper_terminal']==records[keeper],
                'Original keeper failure must remain visible')
        probe=repair['actual_probe'];require(records[probe]==repair['post_accounting'] and
                records[probe][1]=='Kimi-C1-Delta-delayed-post-r312b' and records[probe][2:4]==['COMPLETED','0:0'],
                'Independent delayed probe accounting differs')
        post=repair['rows']
        require(all(v['passed'] and v['CPU_only'] and not v['GPU_runtime_initialized'] and
                v['control30040_LISTEN']==0 and v['actual_step']==probe for v in post),'Recovery resource proof incomplete')
        refs=[{'node':i,'path':str(REMOTE/'results'/(LABEL+'-'+actual+'-node'+str(i)+'.json'))}for i in range(2)]
    require(len(post)==2 and {v['node']for v in post}=={'a14r2n08','a14r2n09'} and
            all(v['KFD_empty'] and v['control30040_bindable'] for v in post),'Postcheck incomplete')
    require(len(refs)==2 and {v['node']for v in refs}=={0,1},'Both node reports required')
    workers=[]
    for ref in refs:
        relative=PurePosixPath(ref['path']).relative_to(REMOTE);q=read(str(relative),ref.get('SHA256'))
        require(q['actual_GPU']==actual and q['source_manifest_SHA256']==MANIFEST and q['CPU_receipt_SHA256']==CPU and
                q['Native_SHA256']==SHIM and q['Main_SHA256']==MAIN,'Owner source differs')
        require(q['C']==q['T']==1 and q['pool']==3 and q['Math']==0 and q['capture']is False and
                q['completed'] and q['normal_close'],'Owner geometry/lifecycle differs')
        bind=q['actual_DSO_bindings'];require(bind['layer0_create_r9038']['DSO_SHA256']==SHIM and
                bind['modern_kimi_KDA_input_counts_r4305']['DSO_SHA256']==MAIN,'Actual DSO binding differs')
        logname='logs/'+LABEL+'-'+actual+'-'+str(ref['node'])+'.log';log=(root/logname).read_bytes()
        require(log.count(('KIMI_C1_DELTA_R311_NORMAL '+str(ref['node'])).encode())==1 and
                b'Using network IBext_v8'in log and b'Using network Socket'not in log,'Network/normal log differs')
        evidence[logname]=hashlib.sha256(log).hexdigest()
        workers.extend(q['workers'])
    result=summarize(workers);result.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='SCOPED_C1_DELTA_COMPONENT_PROFILE_COMPLETE',actual_GPU=actual,CPU_keeper=keeper,
        source_manifest_SHA256=MANIFEST,CPU_receipt_SHA256=CPU,independent_accounting={h:records[h]for h in [actual,keeper]},
        all16_output_state_instrumentation_checks=True,normal_GPU_owner_lifecycle_and_independent_postcheck=True,
        CPU_keeper_clean_exit=repair is None,original_keeper_failure_retained=repair is not None,
        delayed_independent_cleanup_used=cleanup_recovery if repair else None,
        evidence_SHA256=evidence)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--cleanup-recovery',help='Explicit CPU-only delayed receipt; never relabels failed keeper');a=p.parse_args()
    raw=subprocess.check_output(['sacct','-S','2026-10-31','-j','910380,910407','-n','-P','-o',
            'JobID,JobName%120,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30)
    print(json.dumps(analyze(a.root,[x.split('|')for x in raw.splitlines()],a.cleanup_recovery),indent=2,allow_nan=False))
