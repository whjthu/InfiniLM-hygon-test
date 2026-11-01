"""Whole four-request entry with recording Native/control; no model or GPU."""
from pathlib import Path
import importlib.util,json,sys
here=Path(__file__).resolve().parent
parent=here.parent/'Kimi-long-target-publisher-r335'
sys.path.insert(0,str(parent))
spec=importlib.util.spec_from_file_location('recording335',parent/'check_publisher_CPU.py')
recording=importlib.util.module_from_spec(spec);spec.loader.exec_module(recording)
from pair_driver import run_pair,ids_sha,BoundedNative
from grade_output import grade_output


class Control:
    def collect(self,label,value):
        if label.startswith('long-publish:'):
            return [dict(node=n,value=value['value'])for n in range(4)]
        if 'node'in value:return [dict(value,node=n)for n in range(4)]
        return [value.copy()for _ in range(4)]


def fixture(name,width):
    ids=[1]*width
    return dict(name=name,prompt_tokens=ids,input_tokens=width,prompt_ID_SHA256=ids_sha(ids),
        output_budget=128,eos=[163585,163586])


def setup():
    events=[];n=recording.RecordingNative(events);n.control=Control()
    n.calls=[]
    for name in ('call','call_append'):
        original=getattr(n,name)
        def wrapped(*args,_original=original,**kwargs):
            result=_original(*args,**kwargs);n.calls.append(result[-1]);return result
        setattr(n,name,wrapped)
    r=recording.FakeRPC(events)
    f=dict(order=[['retained8635','AR'],['near32k','AR'],['near32k','spec'],['retained8635','spec']],
           fixtures=[fixture('retained8635',3),fixture('near32k',513)])
    return n,r,f,events


def rejects(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid whole entry accepted')


def run():
    checks=[];n,r,f,e=setup();connected=[];saved=[]
    def connect():
        assert len([x for x in saved[-1]['requests'] if x['completed']])==2
        connected.append(True);return r
    def grade(f,ids,finish):
        assert len(ids)==128 and finish=='length'
        return dict(recording=True,not_actual_task_grade=True)
    q=run_pair(n,f,connect,grade,deadline=1e100,cancelled=lambda:False,
               save=lambda v:saved.append(json.loads(json.dumps(v))),enabled=True)
    assert q['completed'] and connected==[True] and len(q['requests'])==4
    assert [x['all_service_cards']for x in q['requests']]==[32,32,33,33]
    assert all(x['normal_reset'] and x['tokens']==128 for x in q['requests'])
    assert r.frames[-1][0]=='close' and q['normal_draft_protocol_close']
    assert all(not x['client_SLO_qualified'] for x in [q])
    checks.append('whole four-request source entry, fixed128 budget, one post-AR draft handshake and protocol close')
    assert all(x['generation_local_wall_s']<=x['request_through_reset_local_wall_s']for x in q['requests'])
    assert all(len(x['emitter_events'])>0 and x['TTFT_s']>=0 for x in q['requests'])
    checks.append('token bursts, TTFT, total through reset, Native records and spec costs retained without client-SLO claim')
    n,r,f,e=setup();rejects(lambda:run_pair(n,f,lambda:r,grade,deadline=0,cancelled=lambda:False,save=lambda v:None,enabled=True));assert not e
    checks.append('expired original deadline stops before reset, forward or draft connect')
    n,r,f,e=setup();rejects(lambda:run_pair(n,f,lambda:r,grade,deadline=1e100,cancelled=lambda:True,save=lambda v:None,enabled=True));assert not e
    checks.append('cancellation stops before first Native action')
    n,r,f,e=setup();states=[]
    def wrong_answer(*args):raise ValueError('recording quality failure')
    rejects(lambda:run_pair(n,f,lambda:r,wrong_answer,deadline=1e100,cancelled=lambda:False,save=lambda v:states.append(json.loads(json.dumps(v))),enabled=True))
    assert len(states[-1]['requests'])==1 and states[-1]['requests'][0]['normal_reset'] and not states[-1]['completed']
    assert not r.frames
    checks.append('answer failure preserves completed arm and reset; stops before second request or draft loading')
    n,r,f,e=setup();f['fixtures'][0]['prompt_ID_SHA256']='0'*64
    rejects(lambda:run_pair(n,f,lambda:r,grade,deadline=1e100,cancelled=lambda:False,save=lambda v:None,enabled=True));assert not e
    checks.append('frozen prompt hash rejected before execution')
    n,r,f,e=setup();rejects(lambda:run_pair(n,f,lambda:r,grade,deadline=1e100,cancelled=lambda:False,save=lambda v:None))
    checks.append('default OFF')
    # The actual existing parser, not a replacement regex, accepts exact integers.
    actual=json.loads((here/'frozen_inputs.json').read_bytes())['fixtures'][0]
    expected=actual['expected_calculate_arguments']
    def text(arguments):
        content=''.join('<|open|>argument key="%s" type="integer"<|sep|>%d<|close|>argument<|sep|>'%(k,v)for k,v in arguments.items())
        return '<|open|>tools<|sep|><|open|>call tool="calculate" index="1"<|sep|>'+content+'<|close|>call<|sep|><|close|>tools<|sep|>'
    result=grade_output(actual,[1,163585],'eos',lambda ids:text(expected))
    assert result['passed'] and result['arguments']==expected
    bad=dict(expected);bad[next(iter(bad))]+=1
    rejects(lambda:grade_output(actual,[1,163585],'eos',lambda ids:text(bad)))
    rejects(lambda:grade_output(actual,[1],'length',lambda ids:text(expected)))
    checks.append('actual pinned XTML parser: exact tool arguments pass; wrong integer and length cap reject')
    return dict(passed=True,checks=checks,count=len(checks),recording_Native=True,
                actual_Native_Context=False,GPU=False,checkpoint_loaded=False,
                production_frozen_inputs_used_in_timing_test=False,performance_gain='UNKNOWN')


if __name__=='__main__': print(json.dumps(run()))
