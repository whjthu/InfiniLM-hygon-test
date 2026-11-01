"""Read-only eight-block ABBA audit of one current-shape Delta allocation candidate."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,math,statistics,subprocess
LABEL='Kimi-C1-gate-alias-GPU-r318'
MANIFEST='161d47e3a0d78d2aa924134804bd08d8088e5df792a0c9b6cfd8c0a67b8c57b7'
CPU='a1fcbd6e0a6f1f8ca2f5ded1ee118cf55a84c1a5b54198c93022bb5f164c1a83'
SHIM='aa8bb59260fd6a7e3344f58583201929593a797ea45463429215114b53b60b6d'
MAIN='eee91fa56997d862c90ae7f4b0ec3944ef5440f60b08d3f552cadf4447622ba3'
REMOTE=PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
PLAN=[('OFF',-1,True),('ON',-1,True)]+[(arm,b,False)for b in range(8)for arm in ['OFF','ON','ON','OFF']]


def require(ok,message):
    if not ok:raise ValueError(message)


def summarize(workers):
    require(len(workers)==16 and sorted(x['rank']for x in workers)==list(range(16)),'Unique16owners required')
    errors=[]
    for owner in workers:
        require(owner['completed'] and owner['normal_close'] and owner['layer_calls']==35 and 'error'not in owner,'Owner incomplete')
        samples=owner['samples'];require(len(samples)==34,'Incomplete samples')
        require([(x['arm'],x['block'],x['warmup'])for x in samples]==PLAN,'Wrong ABBA order')
        require([x['sample_index']for x in samples]==list(range(34)),'Wrong sample index')
        for x in samples:
            require(x['finite']is True and x['source_rows_unchanged']is True and x['profile_enabled']is False,
                    'Output/state/profile invariant failed')
            require(type(x['API_wall_ms'])in(int,float) and math.isfinite(x['API_wall_ms']) and x['API_wall_ms']>0,'Bad timing')
            require(x['actual_alias_counter_delta']==[1,int(x['arm']=='ON'),int(x['arm']=='OFF')],'Arm not actually selected')
            diag=x['numeric_diagnostics'];require(set(diag)=={'output','conv_write_row2','SSM_write_row2'},'Wrong numeric scope')
            for name,d in diag.items():
                require(math.isfinite(d['relative_RMS']) and 0<=d['relative_RMS']<=.02 and
                        math.isfinite(d['max_abs']) and d['max_abs']>=0,'Numerical regression')
                errors.append((name,d['relative_RMS'],d['max_abs']))
    # Full TP layer completion: use same indexed sample's maximum among all16
    # owners. Never sum owner times or choose fastest ranks independently.
    times=[max(owner['samples'][i]['API_wall_ms']for owner in workers)for i in range(34)]
    blocks=[]
    for b in range(8):
        a,x,y,d=times[2+4*b:6+4*b]
        logratio=(math.log(a)+math.log(d)-math.log(x)-math.log(y))/2
        blocks.append({'block':b,'OFF_ms':[a,d],'ON_ms':[x,y],
            'OFF_over_ON_geometric_ratio':math.exp(logratio),
            'OFF_relative_spread':abs(a-d)/math.sqrt(a*d),'ON_relative_spread':abs(x-y)/math.sqrt(x*y)})
    ratios=[b['OFF_over_ON_geometric_ratio']for b in blocks]
    gain=math.exp(statistics.mean(math.log(x)for x in ratios))
    maxspread=max(b['OFF_relative_spread']for b in blocks)
    return {'scope':'One TP16 Delta component; historicalnormalized512prefix+T1, pool3/Math0, nofullModel/live32K replay',
            'warmups_excluded':2,'ABBA_blocks':8,'measured_calls_per_owner':32,'profile_enabled':False,
            'blocks':blocks,'geometric_OFF_over_ON_ratio':gain,'candidate_time_reduction_percent':100*(1-1/gain),
            'blocks_faster_ON':sum(x>1 for x in ratios),'block_ratio_min':min(ratios),'block_ratio_max':max(ratios),
            'max_OFF_relative_spread':maxspread,'point_gain_exceeds_largest_OFF_pair_spread':gain-1>maxspread,
            'all_blocks_faster_ON':all(x>1 for x in ratios),
            'numeric_maxima':{n:{'relative_RMS':max(r for k,r,a in errors if k==n),'max_abs':max(a for k,r,a in errors if k==n)}for n in ['output','conv_write_row2','SSM_write_row2']},
            'statistical_significance_claim':False,'automatic_adoption':False,'Model_gain_qualified':False,
            'limits':['APIwall includes original input/output transfers and final drain; not pure kernel.',
                      'Eight fixed-order ABBA blocks are descriptive; nonlinear drift/contention remains possible.',
                      'One1536B allocation removed whenselected, not observed HBMtraffic or peakmemory saving.',
                      'Component result cannot establish current32KModel/userTPS, quality or clientSLO.']}


def analyze(root,accounting):
    root=Path(root).resolve();evidence={}
    def read(name,expected=None):
        p=(root/name).resolve();require(p.is_relative_to(root),'Evidence escapes root')
        raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();require(expected is None or h==expected,'Evidence SHA differs')
        evidence[str(name)]=h;return json.loads(raw)
    w=read('results/'+LABEL+'-watch.json')
    require(w['completed'] and w['component_lifecycle_qualified'] and w['post_empty'] and w['one_attempt'],'Incomplete terminal audit')
    require(w['source_manifest_SHA256']==MANIFEST and w['CPU_receipt_SHA256']==CPU,'Unexpected source/CPU')
    actual,keeper=w['actual_GPU'],w['CPU_keeper'];require(w['terminal'][0]==actual and w['terminal'][2:4]==['COMPLETED','0:0'],'GPU not completed0')
    records={x[0]:x for x in accounting}
    require(all(h in records and records[h][2:4]==['COMPLETED','0:0']for h in [actual,keeper]),'Independent accounting incomplete')
    require(len(w['post'])==2 and {v['node']for v in w['post']}=={'a14r2n08','a14r2n09'} and
        all(v['passed'] and v['CPU_only'] and not v['GPU_runtime_initialized'] and v['KFD_empty'] and
            v['control30040_bindable'] and v['control30040_LISTEN']==0 for v in w['post']),'Independent postcheck incomplete')
    refs=w['owners'];require(len(refs)==2 and {x['node']for x in refs}=={0,1},'Both owner reports required')
    workers=[]
    for ref in refs:
        name=PurePosixPath(ref['path']).relative_to(REMOTE);q=read(str(name),ref['SHA256'])
        require(q['actual_GPU']==actual and q['source_manifest_SHA256']==MANIFEST and q['CPU_receipt_SHA256']==CPU and
                q['Native_SHA256']==SHIM and q['Main_SHA256']==MAIN,'Owner source mismatch')
        require(q['completed'] and q['normal_close'] and q['owners']==8 and q['C']==q['T']==1 and
                q['pool']==3 and q['Math']==0 and q['capture']is False and q['profile_enabled']is False and q['ABBA_blocks']==8,
                'Owner scope/lifecycle mismatch')
        bind=q['actual_DSO_bindings'];require(bind['layer0_create_r9038']['DSO_SHA256']==SHIM and
                bind['modern_kimi_KDA_input_counts_r4305']['DSO_SHA256']==MAIN,'Actual loaded DSO differs')
        logname='logs/'+LABEL+'-'+actual+'-'+str(ref['node'])+'.log';raw=(root/logname).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==ref['log_SHA256'] and
                raw.count(('KIMI_C1_DELTA_R318_NORMAL '+str(ref['node'])).encode())==1 and
                b'Using network IBext_v8'in raw and b'Using network Socket'not in raw,'Normal/network log proof differs')
        evidence[logname]=ref['log_SHA256'];workers.extend(q['workers'])
    result=summarize(workers);result.update(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='BOUNDED_COMPONENT_ABBA_CONSUMED_NO_AUTOMATIC_MODEL_PROMOTION',actual_GPU=actual,CPU_keeper=keeper,
        source_manifest_SHA256=MANIFEST,CPU_receipt_SHA256=CPU,independent_accounting={h:records[h]for h in [actual,keeper]},
        all16_alias_hits_numeric_source_rows_and_lifecycle_checked=True,independent_KFD_ports_clean=True,
        evidence_SHA256=evidence)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    raw=subprocess.check_output(['sacct','-S','2026-10-31','-j','910380,910407','-n','-P',
        '-o','JobID,JobName%120,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=30)
    print(json.dumps(analyze(a.root,[x.split('|')for x in raw.splitlines()]),indent=2,allow_nan=False))
