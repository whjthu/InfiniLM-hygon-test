"""CPU contract fixtures. Artificial clocks are never performance results."""
import json
from online_service_core_r8451 import Scheduler
from online_service_metrics_r8451 import evaluate


def reject(f):
    try:
        f()
    except (AssertionError, KeyError):
        return
    raise AssertionError('invalid transition accepted')


s = Scheduler(stages=2,micro=4,max_users=8,max_inflight=2,rows=8,eos=[2])
for i in range(4):
    s.arrive(str(i), [10,11], 3, 0)
j0 = s.dispatch(0)
assert j0.keys == ('0','1','2','3')
assert s.take(0, .01) == j0
reject(lambda:s.take(0,.02))
s.finish(0,0,.02)
for i in range(4,8):
    s.arrive(str(i),[12,13,14],2,.03)
j1=s.dispatch(.03)
assert j1.pasts == (0,0,0,0) and j1.rows == (5,6,7,8)
assert s.dispatch(.04) is None
assert s.take(0,.04)==j1
s.finish(0,1,.05)
assert s.take(1,.05)==j0
s.finish(1,0,.06,ids=[20]*4)
# First cohort re-enters immediately while the second remains in flight.
j2=s.dispatch(.06)
assert j2.kind=='decode' and j2.rows==j0.rows and j2.pasts==(2,)*4
assert s.take(1,.07)==j1
s.finish(1,1,.08,ids=[21]*4)
assert s.take(0,.09)==j2
s.finish(0,2,.10)
j3=s.dispatch(.10)
assert j3.rows==j1.rows and j3.pasts==(3,)*4
assert s.take(0,.11)==j3
s.finish(0,3,.12)
assert s.take(1,.13)==j2
s.finish(1,2,.14,ids=[2,22,22,22])
j4=s.dispatch(.14)
assert j4.rows==(2,3,4) and j4.pasts==(3,3,3) # EOS row leaves, survivors regroup.
assert s.take(0,.15)==j4
s.finish(0,4,.16)
assert s.take(1,.17)==j3
s.finish(1,3,.18,ids=[23]*4)
assert s.take(1,.19)==j4
s.finish(1,4,.20,ids=[24]*3)
s.arrive('8',[15,16],1,.21)
assert s.dispatch(.21) is None # completed rows cannot be silently reused.
assert s.can_reset_epoch()
reject(lambda:s.reset_epoch_after_native_world_drain(.21,owners_reset_verified=False))
s.reset_epoch_after_native_world_drain(.22,owners_reset_verified=True)
j5=s.dispatch(.22)
assert j5.rows==(1,) and j5.pasts==(0,)
assert s.take(0,.23)==j5
s.finish(0,5,.24)
assert s.take(1,.25)==j5
s.finish(1,5,.26,ids=[25])
assert s.can_reset_epoch()
reject(lambda:s.finish(1,5,.27,ids=[25]))
metrics=evaluate(dict(events=s.events,window=[0,.26],cards=16,boundary='emitter'))
assert metrics['statuses']['completed']==9
assert metrics['client_delivery']=='NOT_MEASURED'
assert metrics['requests'][0]['user_decode_tok_s']==0 # EOS is not useful output.
assert metrics['requests'][-1]['user_decode_tok_s'] is None
assert metrics['requests'][-1]['queue_to_admit_s'] > 0
assert metrics['aggregate_decode_tok_s']==10/.26
assert metrics['mean_queued_requests']>0
assert metrics['mean_admitted_requests']>0
bad=json.loads(json.dumps(dict(events=s.events,window=[0,.26],cards=16,boundary='client')))
reject(lambda:evaluate(bad))
bad['boundary']='emitter'
next(e for e in bad['events'] if e['kind']=='token')['pad']=True
reject(lambda:evaluate(bad))
reject(lambda:s.arrive('0',[2],1,.3))
s.fail(.3,'synthetic failure fixture')
reject(lambda:s.dispatch(.3))
print(json.dumps(dict(CPU_contract_passed=True,synthetic_fixture_only=True,
    actual_GPU_or_Model_performance='NOT_RUN',checks=[
    'ordered PP jobs; no simultaneous owner calls',
    'new arrivals while prior cohort is in flight',
    'emitter-ready decode reentry before other cohort finishes',
    'EOS survivor regrouping; exact physical row/past',
    'row exhaustion waits for verified native world reset',
    'reject duplicate request and stale completion',
    'reject pad numerator and fabricated client boundary',
    'queue occupancy and single-token undefined speed',
    'failure latch prevents new work'])) )
