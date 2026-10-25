import pathlib,json,random,collections,time,hashlib
from policy import greedy,rejection,ReplayStateModel,sample
start=time.perf_counter();checks=[]
for a in range(8):
    props=tuple(range(10,17));top=list(props)+[77]
    if a<7:top[a]=77
    d=greedy(9,props,top);assert d.accepted==a and d.emitted==props[:a]+(77,)
    s=ReplayStateModel((1,2));s.verify((9,)+props);s.commit(d);assert s.active==(1,2,9)+props[:a]and d.pending_anchor==77
    e=greedy(77,(),(88,));s.verify((77,));s.commit(e);assert s.active[-1]==77
    checks.append({'accepted':a,'replay':d.needs_replay,'commit_length':len(d.commit_inputs),'pending_once':True})
for i in range(4):
    d=greedy(9,(10,11,12),(10,11,12,13),eos=((10,11,12,13)[i],));assert d.terminal and len(d.emitted)==i+1 and not d.needs_replay
for n in range(1,5):
    d=greedy(9,(10,11,12),(10,11,12,13),remaining=n);assert d.terminal and len(d.emitted)==n
for p,q in [((.05,.15,.8),(.6,.4,0)),((0,1,0),(.5,0,.5)),((.2,.3,.5),(.2,.3,.5))]:
    accept=[min(a,b)for a,b in zip(p,q)];reject=1-sum(accept);res=[max(a-b,0)for a,b in zip(p,q)];mass=[accept[i]+(reject*res[i]/sum(res)if sum(res)>0 else 0)for i in range(3)];assert max(abs(a-b)for a,b in zip(mass,p))<1e-12
P=[(.7,.2,.1),(.2,.5,.3),(.05,.15,.8)];Q=[(.3,.7,0),(.6,0,.4),(.2,.2,.6)]
def target(prefix):return P[prefix[-1]]if prefix else(.1,.3,.6)
def draft(prefix):return Q[prefix[-1]]if prefix else(.6,.4,0)
rng=random.Random(9046);N=60000;counts=collections.Counter()
for _ in range(N):
    out=[]
    while len(out)<2:
        y=sample(draft(out),rng);d=rejection(99,(y,),(target(out),target(out+[y])),(draft(out),),rng,remaining=2-len(out));out.extend(d.emitted)
    counts[tuple(out)]+=1
expected={(a,b):target(())[a]*target((a,))[b]for a in range(3)for b in range(3)};error=max(abs(counts[k]/N-v)for k,v in expected.items());assert error<.012
report={'status':'CPU_POLICY_COMMIT_AND_SAMPLING_DISTRIBUTION_PASS','GPU':False,'native_KDA_numeric':'NOT_SIMULATED','greedy_prefixes':checks,'EOS_cases':4,'length_cap_cases':4,'analytic_probability_cases':3,'stochastic_sequences':N,'two_token_joint_expected':{str(k):v for k,v in expected.items()},'two_token_joint_empirical':{str(k):counts[k]/N for k in expected},'maximum_absolute_probability_error':error,'same_seed_or_AR_sequence_identity_required':False,'sampling_requires_real_conditional_q':True,'elapsed_s':time.perf_counter()-start,'sources':{n:hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()for n in ['policy.py','verify_cpu.py']}}
pathlib.Path('CPU-policy-receipt-r9046.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
