"""Frozen constructor-only cache policy. Default remains all-NoV."""
from dataclasses import dataclass,asdict
BOUNDS=(0,38,78)
VALUE_BYTES=32768*8*256*2
DEFAULT='nov-all'
MIXED='stage0-fullV-stage1-NoV'
FLAGS={DEFAULT:(1,1),MIXED:(0,1)}
@dataclass(frozen=True)
class StagePolicy:
    policy:str
    stage:int
    noV_flag:int
    layer_begin:int
    layer_end:int
    exact_bank_count:int
    projected_value_bytes:int
    def metadata(self):return asdict(self)
def stage_policy(name,stage):
    if type(name)is not str or name not in FLAGS or type(stage)is not int or stage not in(0,1):raise ValueError('Exact constructor cache policy/stage required')
    flag=FLAGS[name][stage];n=BOUNDS[stage+1]-BOUNDS[stage]
    return StagePolicy(name,stage,flag,BOUNDS[stage],BOUNDS[stage+1],(3+(1-flag))*n,(1-flag)*n*VALUE_BYTES)
def validate_config(c):
    name=c.get('cache_policy_r452',DEFAULT)
    if (c['TP'],c['PP'],c['rows'],c['context'])!=(8,2,1,32768):raise ValueError('C1 TP8 PP2 full32768 only')
    flags=c.get('cache_stage_flags_r452',list(FLAGS.get(name,())))
    if type(flags)is not list or any(type(v)is not int for v in flags)or tuple(flags)!=FLAGS.get(name):raise ValueError('Both stage flags must match typed policy')
    prefix='glm-mixedV16-w'if name==MIXED else'glm-nov16-w'
    if c['profile']not in[prefix+'2',prefix+'4']:raise ValueError('Cache/profile identity mismatch')
    backend='private-native-glm-mixedV16'if name==MIXED else'qualified-native-glm-nov16'
    if c['backend']!=backend:raise ValueError('Cache/backend identity mismatch')
    return name
def for_config(c,stage):return stage_policy(validate_config(c),stage)
def from_environment(env,stage):
    p=stage_policy(env.get('GLM_CACHE_POLICY_R452',DEFAULT),stage)
    if env['GLM_V_ON_DEMAND_R403']!=str(p.noV_flag):raise ValueError('Constructor environment/cache policy disagree')
    return p
def global_metadata(c):
    name=validate_config(c)
    return dict(policy=name,stage_flags=list(FLAGS[name]),stages=[stage_policy(name,s).metadata()for s in range(2)],constructor_only=True,request_switch=False)
def validate_census(c,p):
    if[c['layer_begin'],c['layer_end'],c['rows'],c['capacity'],c['loaded']]!=[p.layer_begin,p.layer_end,1,32768,False]:raise RuntimeError('C1PP2 unloaded census mismatch')
    if[c['layer']for c in c['cache_layers']]!=list(range(p.layer_begin,p.layer_end)):raise RuntimeError('Missing/duplicate layer census')
    total={}
    for layer in c['cache_layers']:
        kinds={t['category']:t for t in layer['tensors']}
        if len(kinds)!=len(layer['tensors'])or not all(k in kinds for k in['latent','rotary','projected_key']):raise RuntimeError('Required distinct cache tensors missing')
        if ('projected_value'in kinds)!=(p.noV_flag==0):raise RuntimeError('Wrong actual Native V allocation branch')
        shapes={'latent':[1,32768,512],'rotary':[1,32768,64],'projected_key':[1,32768,8,256]}
        if not p.noV_flag:shapes['projected_value']=[1,32768,8,256]
        for k,shape in shapes.items():
            t=kinds[k];n=2
            for v in shape:n*=v
            if t['shape']!=shape or t['bytes']!=n or t['unique_start_address']is not True:raise RuntimeError('Actual cache shape/bytes/unique allocation mismatch')
        for t in kinds.values():total[t['category']]=total.get(t['category'],0)+t['bytes']
    if total!=c['cache_category_bytes']or sum(total.values())!=c['cache_unique_start_bytes']or total.get('projected_value',0)!=p.projected_value_bytes:raise RuntimeError('Actual cache census byte totals mismatch')
    return p.metadata()
