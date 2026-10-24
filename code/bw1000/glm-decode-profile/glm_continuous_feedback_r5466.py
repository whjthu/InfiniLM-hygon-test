"""GLM CPU packet protocol only; no Engine or Runtime calls from handlers."""
import threading,math,statistics
from kimi_cohort_feedback_idle_r4915 import FeedbackHub as ParentHub,FeedbackClient,decode_cohorts

class FeedbackHub(ParentHub):
    def fail_case(self,message):
        with self.cv:
            super().fail_case(message)
            if self.failure is not None:
                self.pending.clear()
            self.cv.notify_all()

    def dispatch(self,rank,req):
        if req.get('op')=='abort':
            with self.cv:
                assert req.get('case')==self.case and self.case is not None
                message=req.get('message')
                assert isinstance(message,str)and 0<len(message)<=1000
                self.fail_case('owner %d: %s'%(rank,message))
                return None
        return super().dispatch(rank,req)

class CheckedFeedback:
    def __init__(self,client,tail_nonemitter=False):
        self.client=client;self.tail_nonemitter=tail_nonemitter
        self.expected={};self.owner=threading.get_ident();self.in_forward=False

    def request(self,op,case,**values):
        assert threading.get_ident()==self.owner and not self.in_forward
        packet=self.client.request(op,case,**values)
        if op=='take'and self.tail_nonemitter:
            key=(values['cohort'],values['step'])
            rows,ids=self.expected.pop(key)
            if packet['rows']!=rows or packet['ids']!=ids:
                raise RuntimeError('tail owner argmax differs from emitter packet')
        return packet

    def forward(self,callback,rows,step,cohort,is_emitter):
        assert threading.get_ident()==self.owner and not self.in_forward
        self.in_forward=True
        try:
            # Callback owns all host/device arguments through synchronous return.
            selected,seconds=callback(rows,step,cohort)
            if self.tail_nonemitter:
                assert selected is not None
                self.expected[(cohort,step)]=(list(rows),list(selected))
            return(selected if is_emitter else None,seconds)
        finally:self.in_forward=False

def safe_decode(*,feedback,case,is_emitter,forward,**kwargs):
    try:
        samples=decode_cohorts(feedback=feedback,case=case,is_emitter=is_emitter,
            forward=lambda rows,step,cohort:feedback.forward(forward,rows,step,cohort,is_emitter),**kwargs)
        assert not feedback.expected and not feedback.in_forward
        return samples
    except BaseException as error:
        try:feedback.client.request('abort',case,message=(type(error).__name__+': '+str(error))[:1000])
        except BaseException:feedback.client.close()
        raise

def tail_ready_metrics(generated,completions,eos,elapsed,cards,limit):
    assert generated and len(generated)==len(completions)and elapsed>0 and cards==16
    for ids,times in zip(generated,completions):
        assert ids and len(ids)==len(times)and 1<=len(ids)<=limit
        assert ids[-1]in eos or len(ids)==limit
        assert not any(v in eos for v in ids[:-1])
        assert all(math.isfinite(t)and t>=0 for t in times)and times==sorted(times)
    anchor=max(t[0]for t in completions);terminal=max(t[-1]for t in completions)
    useful=[sum(v not in eos for v in ids)for ids in generated]
    decode=sum(sum(v not in eos and t>anchor for v,t in zip(ids,times))for ids,times in zip(generated,completions))
    rate=decode/(terminal-anchor)if terminal>anchor else 0.0
    perrequest=[]
    for ids,times in zip(generated,completions):
        n=sum(v not in eos for v in ids[1:]);duration=times[-1]-times[0]
        perrequest.append(n/duration if duration>0 else 0.0)
    assert terminal>0 or sum(useful)==0,'nonzero useful tokens require positive tail interval'
    tail_e2e=sum(useful)/terminal if terminal>0 else 0.0
    return {'timing_observer':'tailrank0 local ready stamp after same argmax/raw observer, before feedback; no HTTP',
        'useful_output_tokens':sum(useful),'decode_interval_useful_tokens':decode,
        'all_prefill_complete_seconds':anchor,'last_completion_seconds':terminal,
        'aggregate_useful_decode_tokens_s':rate,'decode_tokens_s_per_card':rate/cards,
        'including_prefill_useful_tokens_s':tail_e2e,'including_prefill_tokens_s_per_card':tail_e2e/cards,
        'including_prefill_endpoint':'tailrank0 phase start to last decision ready, before feedback in both arms',
        'secondary_phase_elapsed_seconds':elapsed,'secondary_phase_E2E_tokens_s':sum(useful)/elapsed,
        'request_useful_decode_tokens_s':perrequest,'mean_request_useful_decode_tokens_s':statistics.mean(perrequest),
        'TTFT_seconds_per_request':[t[0]for t in completions],'terminal_requests':len(generated),
        'first_token_EOS_requests':sum(ids[0]in eos for ids in generated),'EOS_terminal_requests':sum(ids[-1]in eos for ids in generated),
        'all_serving_cards':cards,'no_forward_after_EOS':True}
