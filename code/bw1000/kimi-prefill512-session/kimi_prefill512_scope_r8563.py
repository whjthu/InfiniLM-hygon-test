"""Serial owner scope only; geometry comes from the immutable actual job.

One Python process owns one PP stage. Its main thread calls the synchronous
Engine entry; socket/producer threads never call Engine or modify these flags.
Phase mode changes occur after drained/reset all-owner control barriers.
"""
import os,threading
from online_kimi_provider_r8517 import KimiLoadedAdapter

MODE='INFINILM_KIMI_PREFILL512_R8562'
ELIGIBLE='INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562'

def eligible_job(job):
    pasts,inputs=job['pasts'],job['inputs']
    return (job['kind']=='prefill' and 0<len(inputs)==len(pasts)
            and all(type(p)is int and p==0 for p in pasts)
            and all(len(row)==512 for row in inputs)
            and sum(map(len,inputs))<=8192)

class ScopedProvider(KimiLoadedAdapter):
    def forward(self,job):
        assert threading.current_thread() is threading.main_thread()
        assert os.environ.get(MODE,'0') in ('0','1')
        old=os.environ.get(ELIGIBLE)
        os.environ[ELIGIBLE]='1' if eligible_job(job) else '0'
        try:return super().forward(job)
        finally:
            if old is None:os.environ.pop(ELIGIBLE,None)
            else:os.environ[ELIGIBLE]=old
    def close(self):self.closed=True # old loader's outer finally destroys Engine.
