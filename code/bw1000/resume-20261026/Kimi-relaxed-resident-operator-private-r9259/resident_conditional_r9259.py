"""Data-only bounded conditional admission for the already-running frozen r9258b.
No model/native import, subprocess model launch, GPU API, or parameter tuning.
"""
import pathlib,json,os,time,socket,subprocess,hashlib,datetime,traceback,sys,statistics
ROOT=pathlib.Path.home()/'hygon-inference-refactor-20261014'
MODEL=ROOT/'incoming/Kimi-relaxed-hostlogits-r9258b'
sys.path.insert(0,str(MODEL/'runtime'))
from quality_gate_r9258 import paired_gate
RUN='kimi-relaxed-A-r9258b-04e27169'
TARGET='910379.466';DRAFT='910371.580'
QUEUE=ROOT/'results/Kimi-relaxed-A-queue-r9258'
HEALTH=ROOT/'results/Kimi-relaxed-conditional-r9259-health.json'
EMITTER=ROOT/'results'/('Kimi-relaxed-A-r9258-'+RUN+'-node2.json')
PROGRESS=ROOT/'results'/('Kimi-relaxed-A-r9258-'+RUN+'-node0.progress.json')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(p):
    for n in range(5):
        try:return json.loads(p.read_bytes())
        except(OSError,ValueError):
            if n==4:raise
            time.sleep(.2)
def scalar_case(c):
    users=[]
    for u in c['request_metrics']:
        q=u['quality']
        users.append({k:u.get(k)for k in ['useful_tokens','useful_decode_tokens','first_ready_monotonic','last_ready_monotonic','completion_monotonic','native_TTFT_s','user_firstlast_tok_s','ITL_P50_s','ITL_P95_s','ITL_P99_s','ITL_max_s','EOS','truncated']}|{'quality':{k:q.get(k)for k in ['id','task','domain','sample_index','expected','extracted','correct']}})
    assert len(users)==1 and c['configured_C']==1 and c['all_serving_GPU_cards']==33
    cycles=c['cycles']
    return {k:c.get(k)for k in ['label','relaxed_gap_epsilon','total_useful_decode_tokens','decode_common_window_s','aggregate_decode_tok_s','aggregate_decode_tok_s_per_card','prefill_all_complete_s','admission_native_monotonic']}|{
        'user':users[0],'command':c['command'],'allcards':33,'cycles':len(cycles),
        'cost_s':{k:sum(z.get(k,0)for z in cycles)for k in ['verify_CAPI_maxhost_s','draft_RPC_group_s','replay_CAPI_maxhost_s','cycle_local_wall_s']},
        'gap_gather_HOST_s':sum((z.get('relaxed_gap')or{}).get('gather_wall_s',0)for z in cycles),
        'verified_relaxed':sum(z.get('verified_relaxed',0)for z in cycles),'emitted_relaxed':sum(z.get('emitted_relaxed',0)for z in cycles),
        'accepted':sum(z['accepted_proposals']for u in c['request_metrics']for z in u['cycles']),
        'proposed':sum(z['proposed']for u in c['request_metrics']for z in u['cycles'])}
def economics(cases):
    by={0.:[],.10:[]}
    for c in cases:by[c['relaxed_gap_epsilon']].append(c)
    quality={}
    arm={}
    for eps,cs in by.items():
        qs=[];N=sum(c['total_useful_decode_tokens']for c in cs);wall=sum(c['decode_common_window_s']for c in cs)
        for c in cs:
            u=c['user'];q=u['quality'];qs.append({'id':q['id'],'domain':q['domain'],'status':'completed'if u['EOS']or u['truncated']else'unfinished','correct':bool(q['correct'])})
        quality[eps]=qs
        arm[str(eps)]={'n':len(cs),'tokens':N,'decode_common_s':wall,'decode_tok_s':N/wall if wall else None,'per33card':N/wall/33 if wall else None,
                      'correct':sum(c['user']['quality']['correct']for c in cs),'EOS':sum(c['user']['EOS']for c in cs),'caps':sum(c['user']['truncated']for c in cs),
                      'parsefail':sum(c['user']['quality']['extracted']is None for c in cs),
                      'correct_naturalEOS_requests':sum(c['user']['EOS']and c['user']['quality']['correct']for c in cs),
                      'correct_naturalEOS_request_goodput_s':sum(c['user']['EOS']and c['user']['quality']['correct']for c in cs)/wall if wall else None,
                      'correct_naturalEOS_decode_tok_s':sum(c['total_useful_decode_tokens']for c in cs if c['user']['EOS']and c['user']['quality']['correct'])/wall if wall else None,
                      'correct_naturalEOS_GT15_decode_tok_s_per33card':sum(c['total_useful_decode_tokens']for c in cs if c['user']['EOS']and c['user']['quality']['correct']and(c['user']['user_firstlast_tok_s']or 0)>15)/wall/33 if wall else None,
                      'peruser_rates':[c['user']['user_firstlast_tok_s']for c in cs],
                      'task_completion_s':[c['user']['completion_monotonic']-c['admission_native_monotonic']for c in cs],
                      'emitted_relaxed':sum(c['emitted_relaxed']for c in cs)}
    gate=paired_gate(quality[0.],quality[.10])
    a={c['user']['quality']['id']:c for c in by[0.]};b={c['user']['quality']['id']:c for c in by[.10]}
    normalized=[]
    for key in sorted(a):
        c0,c1=a[key],b[key];n0,n1=c0['total_useful_decode_tokens'],c1['total_useful_decode_tokens']
        normalized.append({'id':key,'strict_tokens':n0,'candidate_tokens':n1,'strict_common_s':c0['decode_common_window_s'],'candidate_common_s':c1['decode_common_window_s'],
            'candidate_common_s_scaled_to_strict_length':c1['decode_common_window_s']*n0/n1 if n1 else None,
            'scope':'descriptive tokenlength adjustment, not observed fixedlength latency'})
    norm=sum(x['candidate_common_s_scaled_to_strict_length']or float('inf')for x in normalized)
    ratio=sum(x['strict_common_s']for x in normalized)/norm if norm else 0
    cap_anomaly=any(c['user']['truncated']for c in cases)
    positive=not gate['stop']and not cap_anomaly and arm['0.1']['decode_tok_s']>arm['0.0']['decode_tok_s']and ratio>1 and arm['0.1']['emitted_relaxed']>0
    return {'arms':arm,'quality_stop':gate,'any_cap_anomaly':cap_anomaly,'length_adjusted_ratio':ratio,'length_adjusted_cases':normalized,'positive_for_conditional_expansion':positive,'quality_noninferiority_claim':False,'rates_decode_only':True}
def selftest():
    c={'relaxed_gap_epsilon':0.,'total_useful_decode_tokens':100,'decode_common_window_s':10.,'admission_native_monotonic':0.,'emitted_relaxed':0,'user':{'quality':{'id':'x','domain':'a','correct':True,'extracted':'1'},'EOS':True,'truncated':False,'user_firstlast_tok_s':10.,'completion_monotonic':11.}}
    b=json.loads(json.dumps(c));b.update(relaxed_gap_epsilon=.10,decode_common_window_s=8.,emitted_relaxed=1)
    assert economics([c,b])['positive_for_conditional_expansion']
    b['user']['quality']['correct']=False;assert economics([c,b])['quality_stop']['stop']
    b['user']['quality']['correct']=True;b['user']['truncated']=True;assert not economics([c,b])['positive_for_conditional_expansion']
    b['user']['truncated']=False;b['decode_common_window_s']=12.;assert not economics([c,b])['positive_for_conditional_expansion']
    return{'passed':True,'CPU_only':True,'GPU':False,'gates':'fixed .10, >2pp overall/>5pp domain stop, caps stop, positive actual+descriptive lengthnormalized decode before expansion'}
def main():
    assert selftest()['passed']
    source_index=load(MODEL/'producer-CPU-qualified-r9258b.json')['source_files_SHA256']
    for name in ['conditional-repeat4-r9258.json','conditional-heldout95-commands-r9258b.json','admission-ledger-r9258b.json']:
        assert sha((MODEL/name).read_bytes())==source_index[name],name
    assert not HEALTH.exists()
    w={'schema':1,'actual_CPU_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'PID':os.getpid(),'run':RUN,'GPU_submit':False,'initial_expected':[],'published_pairs':[],'completed':False,'pair_ahead_limit':1,'source_SHA256':sha(pathlib.Path(__file__).read_bytes())}
    env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))}
    def save():
        w['heartbeatUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        raw=(json.dumps(w,indent=2)+'\n').encode();tmp=HEALTH.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(HEALTH)
    def account():
        out=subprocess.check_output(['sacct','-S','2026-10-26','-j',TARGET+','+DRAFT,'-n','-P','-o','JobIDRaw,State,ExitCode,ElapsedRaw'],env=env,text=True,timeout=25)
        rows=[x.split('|')for x in out.splitlines()if x.split('|')[0]in[TARGET,DRAFT]];assert len(rows)==2;return rows
    def publish(commands,seq):
        published=[]
        for j,c in enumerate(commands):
            assert c['C']==1 and c['W']==8 and c['cap']==512 and c['strategy']=='state'and c['relaxed_gap_epsilon']in[0,.10]and not c['draft_projected_cache']
            assert all(x['id']!='mmlu-test-11704'for x in c['requests'])
            raw=(json.dumps(c,ensure_ascii=False,indent=2)+'\n').encode();p=QUEUE/('q%04d-conditional-r9259.json'%(seq+j))
            if p.exists():assert p.read_bytes()==raw
            else:
                tmp=p.with_suffix('.publish');tmp.write_bytes(raw);os.link(tmp,p);tmp.unlink()
            published.append({'name':p.name,'id':c['id'],'SHA256':sha(raw)})
        time.sleep(1.1)
        for p in published:os.utime(QUEUE/p['name'],None)
        os.utime(QUEUE,None);return published
    def stop(reason,done):
        pending=[]
        for p in QUEUE.glob('q*.json'):
            c=load(p)
            if c.get('action')=='case'and c['id']not in done:pending.append(c['id'])
        if pending:
            w['stop_pending_existing_cases']=pending;save();return False
        c={'schema':1,'action':'stop','id':'relaxed-normal-stop-r9259','reason':reason};raw=(json.dumps(c,indent=2)+'\n').encode();p=QUEUE/'q9999-normal-stop-r9259.json'
        if p.exists():assert p.read_bytes()==raw
        else:
            tmp=p.with_suffix('.publish');tmp.write_bytes(raw);os.link(tmp,p);tmp.unlink()
        os.utime(p,None);os.utime(QUEUE,None);w.update(completed=True,normal_stop_published=True,stop_reason=reason);save();return True
    initial={load(p)['id']for p in(MODEL/'pilot-queue').glob('q*.json')};w['initial_expected']=sorted(initial);assert len(initial)==16
    repeat=load(MODEL/'conditional-repeat4-r9258.json')['cases'];held=load(MODEL/'conditional-heldout95-commands-r9258b.json')['paired_cases'];assert len(repeat)==8 and len(held)==190
    allpairs=[repeat[i:i+2]for i in range(0,8,2)]+[held[i:i+2]for i in range(0,190,2)]
    pair=0;waiting=None;stage='pilot';lastfp=None;known={};save();until=time.monotonic()+11600
    try:
        while time.monotonic()<until:
            rows=account();w['same_handles_sacct']=rows
            if any(x[1]in TERMINAL for x in rows):w.update(completed=True,stop_reason='same exact GPU handles terminal; no restart');break
            remaining=10270-next(int(x[3])for x in rows if x[0]==TARGET);w['remaining_autoStop_s']=remaining
            try:progress=load(PROGRESS);w['node0_progress']=progress
            except(OSError,ValueError)as e:w['read_error']=repr(e);save();time.sleep(5);continue
            if progress.get('label')!='resident-idle':save();time.sleep(5);continue
            try:
                st=EMITTER.stat();fp=[st.st_mtime_ns,st.st_size]
                if fp!=lastfp:
                    assert st.st_size<512*1024*1024
                    raw=EMITTER.read_bytes();q=json.loads(raw);assert q['run']==RUN and not q.get('error')and not q.get('shutdown_error')
                    known={c['label']:scalar_case(c)for c in q.get('batch_cases',[])}
                    w['emitter_SHA256']=sha(raw);lastfp=fp;del q,raw
            except(OSError,ValueError)as e:w['read_error']=repr(e);save();time.sleep(3);continue
            done=set(known);w['done_case_ids']=sorted(done)
            if not initial<=done:save();time.sleep(5);continue
            pilot=[known[x]for x in sorted(initial)];w['pilot_economics']=economics(pilot)
            if stage=='pilot':
                if not w['pilot_economics']['positive_for_conditional_expansion']:
                    if stop('pilot no positive task-conditioned decode economics, quality stop, or cap anomaly',done):break
                    time.sleep(3);continue
                stage='repeat4'
            if waiting:
                if not {x['id']for x in waiting}<=done:
                    if time.monotonic()-w.get('last_refresh_mono',0)>10:
                        for x in w['published_pairs'][-1]['files']:os.utime(QUEUE/x['name'],None)
                        os.utime(QUEUE,None);w['last_refresh_mono']=time.monotonic()
                    save();time.sleep(5);continue
                pair+=1;waiting=None
            repeat_ids={x['id']for x in repeat}
            if repeat_ids<=done:
                w['repeat_economics']=economics([known[x]for x in sorted(repeat_ids)])
                if not w['repeat_economics']['positive_for_conditional_expansion']:
                    if stop('reverse4 repeat not positive or quality/cap stop; no heldout expansion',done):break
                    time.sleep(3);continue
                stage='held95'
            actual_held=[known[x['id']]for x in held if x['id']in done]
            # Check only complete same-question pairs; include all cap/parsefail/incomplete statuses in selected pairs.
            counts={}
            for c in actual_held:counts[c['user']['quality']['id']]=counts.get(c['user']['quality']['id'],0)+1
            complete=[c for c in actual_held if counts[c['user']['quality']['id']]==2]
            if complete:
                w['heldout_economics']=economics(complete)
                if w['heldout_economics']['quality_stop']['stop']or w['heldout_economics']['any_cap_anomaly']:
                    if stop('prespecified heldout quality/domain/cap stop',done):break
                    time.sleep(3);continue
            if pair>=len(allpairs):
                if stop('admissible held95 completed, planned96 with one frozen context rejection; analysis done',done):break
                time.sleep(3);continue
            if remaining<1200:
                if stop('insufficient conservative600s x2 pairbudget; preserve cleanup and unruncoverage',done):break
                time.sleep(3);continue
            commands=allpairs[pair]
            if {x['id']for x in commands}<=done:pair+=1;save();continue
            files=publish(commands,100+pair*2);waiting=commands
            w['published_pairs'].append({'pair':pair,'tier':'reverse4'if pair<4 else'held95','files':files,'remaining_s':remaining})
            w['stage']=stage;w['last_refresh_mono']=time.monotonic();save();time.sleep(5)
        else:w['stop_reason']='CPU operator bounded deadline; no model signal or reload'
    except BaseException as e:w['error']=repr(e);w['traceback']=traceback.format_exc();save();raise
    finally:
        w['case_summaries']=[known[x]for x in sorted(known)];w['not_executed_conditional_ids']=[x['id']for pair_ in allpairs for x in pair_ if x['id']not in known];save()
if __name__=='__main__':
    if '--selftest'in sys.argv:print(json.dumps(selftest()))
    else:main()
