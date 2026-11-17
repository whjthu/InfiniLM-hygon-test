"""Controlled synthetic activations, never checkpoint weights or Model activations."""
import array,hashlib,sys
CAPACITY=32768
CASES=((17,1313),(29,1314),(43,8192))
def matrix(seed,columns):
    if seed not in (17,29,43,118,130,144) or columns not in (64,512):raise ValueError('frozen synthetic recipe')
    period=65521
    values=array.array('H',[(0 if i%257==0 else (((i//17+seed)&1)<<15)|(0x3c00+(i*73+seed*997)%0x380)) for i in range(period)])
    if sys.byteorder!='little':values.byteswap()
    block=values.tobytes();n=CAPACITY*columns*2
    return (block*((n+len(block)-1)//len(block)))[:n]
def case(ordinal):
    if type(ordinal) is not int or not 0<=ordinal<3:raise ValueError('three fixed cases only')
    seed,u=CASES[ordinal];ids=array.array('q',range(u))
    if sys.byteorder!='little':ids.byteswap()
    return dict(ordinal=ordinal,seed=seed,U=u,capacity=CAPACITY,latent=matrix(seed,512),rotary=matrix(seed+101,64),ids=ids.tobytes(),kind='controlled synthetic BF16 inputs, legal complete-prefix IDs; not business activations')
def pins(ordinal):
    q=case(ordinal)
    return {**{k:v for k,v in q.items() if k not in ('latent','rotary','ids')},'files':{k:{'bytes':len(q[k]),'SHA256':hashlib.sha256(q[k]).hexdigest()} for k in ('latent','rotary','ids')}}
