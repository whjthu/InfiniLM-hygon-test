"""Controlled component activations: never captured Model traffic or generated tokens."""
import hashlib
import struct
from functools import lru_cache

FULL_CONTEXT=32768
HIDDEN=4096
Q_LORA=1024
H=lambda b:hashlib.sha256(b).hexdigest()

@lru_cache(maxsize=388)
def _row(width,position_mod,variant):
    # All BF16 values are finite, nonzero, signed; periodicity97 crosses C4 groups.
    return struct.pack('<'+'H'*width,*[(0x3e80+((j*17+position_mod*31+variant*43)%128))|
          (0x8000 if (j+position_mod+variant)%3==0 else 0) for j in range(width)])

def inputs(past,tokens,variant=0):
    if type(past)!=int or type(tokens)!=int or variant not in (0,1):
        raise ValueError('exact integer geometry/variant required')
    if not (0<=past<FULL_CONTEXT and 1<=tokens<=512 and past+tokens<=FULL_CONTEXT):
        raise ValueError('complete full32768 geometry required')
    return (b''.join(_row(HIDDEN,p%97,variant) for p in range(past,past+tokens)),
            b''.join(_row(Q_LORA,p%97,variant) for p in range(past,past+tokens)),
            struct.pack('<'+'q'*tokens,*range(past,past+tokens)))

def schedule():
    result=[dict(action='reset',label='initial')]
    def append(past,tokens,label,commit=True,variant=0):
        result.append(dict(action='prepare',past=past,tokens=tokens,variant=variant,
                           label=label,commit=commit,compressed_before=past//4,
                           compressed_after=(past+tokens)//4))
    past=0
    while past<8496:
        n=min(512,8496-past);append(past,n,'history');past+=n
    append(8496,1,'warm_abort',False)
    append(8496,1,'primary');past=8497
    while past<32767:
        n=min(512,32767-past);append(past,n,'history');past+=n
    append(32767,1,'full_boundary')
    result.append(dict(action='reset',label='after_full'))
    append(0,512,'rewrite',variant=1)
    return result

def binding():
    steps=[]
    for s in schedule():
        e=dict(s)
        if e['action']=='prepare':
            values=inputs(e['past'],e['tokens'],e['variant'])
            e['inputs']={k:dict(bytes=len(v),SHA256=H(v)) for k,v in
                         zip(('hidden_BF16','q_lora_BF16','positions_I64'),values)}
        steps.append(e)
    return dict(schema=1,full_context=32768,compressed_capacity=8192,topk=512,
                floor_bytes=16*1024**3,activation_origin='controlled synthetic component inputs; not Model capture',
                steps=steps,prepares=69,commits=68,aborts=1,resets=2,
                submitted_input_rows=33281,committed_input_rows=33280,
                final_past=512,process_order=['OFF','ON','ON','OFF','OFF','ON'],
                paired_units=3,performance_rounds_if_completed=1,
                GPU_GO=False,Model_GO=False)
