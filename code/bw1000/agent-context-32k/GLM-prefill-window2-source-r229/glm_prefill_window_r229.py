"""Source-only dispatcher; no launch integration. All ranks must use one fixed flag.

Default W1 calls the byte-identical r204 driver. W2 is unqualified and must not
be run merely because this local source snapshot exists. One owner/context;
no Native/communicator change, async API or cross-request cache reuse.
"""
import os

def run(*args, **kwargs):
    window=os.environ.get('GLM_PREFILL_CONTROL_WINDOW_R229','1')
    if window not in ('1','2'):
        raise ValueError('GLM_PREFILL_CONTROL_WINDOW_R229 must be exactly 1 or 2')
    if window=='1':
        from glm_agent_service_baseline_r204 import run as selected
    else:
        from glm_agent_service_window2_r229 import run as selected
    return selected(*args, **kwargs)
