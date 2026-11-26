"""Test the full Val5000 denominator and production adapter route with synthetic metadata."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE))
import contract as c
import native


def rejected(fn,*args):
    try:fn(*args)
    except ValueError:return
    raise AssertionError('required rejection')


base_source=REPO/'code/bw1000/gemma-usgs-confirmatory96-native-owner-v2/core'
assert c.sha((base_source/'infer_blind36.py').read_bytes())=='9136cd0aae6278d5eef512aecf720b680ffba5c93e71293d3f2320b7c0b5acda'
assert c.sha((base_source/'STRUCTURAL_BINDING.json').read_bytes())==c.STRUCTURAL
assert c.sha((REPO/'tools/textvqa_effect.py').read_bytes())==c.SCORER
assert c.sha((REPO/'code/bw1000/training-gemma-original-model-v1/MODEL_FILES.json').read_bytes())==native.BASE_MANIFEST
score_record=REPO/'results/bw1000/2026-11-27/training-gemma-usgs-confirmatory96-original3682-first-offline-score.json'
assert c.sha(score_record.read_bytes())=='d36d05412bd96f5437d079fde7c43bf2cf858a8aff69c3b5cbcc1851a8f21fd0'
ids=[dict(question_id=i,image_id=f'{i%3166:016x}') for i in range(5000)]
roster=c.validate_identities(ids)
questions=dict(schema='TEXTVQA_V051_VAL_QUESTIONS_WITHOUT_ANSWERS_V1',annotation_SHA256=c.ANN,
               identity_SHA256=c.IDENTITIES,rows=[dict(r,question='Fixture text?') for r in ids])
media=dict(schema='TEXTVQA_V051_VAL_PRESENT_MEDIA_FOR_INFERENCE_V1',identity_SHA256=c.IDENTITIES,
           annotations_SHA256=c.ANN,original_64_content_overlap=0,
           rows=[dict(image_id=i,relative_path=i+'.jpg',bytes=100,SHA256='1'*64) for i in set(roster.values())])
rows=c.input_plan(roster,questions,media)
assert len(rows)==5000 and len({v[2]['image_id'] for v in rows})==3166
wrong=deepcopy(questions);wrong['rows'][0]['image_id']='f'*16
rejected(c.input_plan,roster,wrong,media)
wrong=deepcopy(questions);wrong['rows'][0]['answers']=['fixture forbidden']
rejected(c.input_plan,roster,wrong,media)
rejected(c.input_plan,roster,questions,dict(media,rows=media['rows'][:-1]))
rejected(c.validate_identities,ids[:-1]+ids[:1])
rejected(c.checkpoint_binding,'base',None)
rejected(c.checkpoint_binding,'Hygon',b'false manifest')
assert c.checkpoint_binding('A100')['weight']==c.A100_WEIGHT


class Parameter:
    _version=0
    def __init__(self, n):self.n=n
    def detach(self):return self
    def cpu(self):return self.n
    def data_ptr(self):return id(self)


class Layer:
    disable_adapters=False
    merged=False
    active_adapters=()
    def __init__(self,name):self.lora_A=self.lora_B={name:object()}
    def forward(self,**_):return 'fixture LORA compute'


class EngineModel:
    training=False
    def __init__(self,name,base):
        self.base=base; self.peft_config={name:object()}
        self.layers={f'block{i}':Layer(name) for i in range(238)}
        self.params={f'block{i}.lora_{letter}.{name}.weight':Parameter(2*i+j)
                     for i in range(238) for j,letter in enumerate('AB')}
    def get_base_model(self):return self.base
    def named_parameters(self):return self.params.items()
    def named_modules(self):return self.layers.items()


ARM='Hygon_before';ADAPTER='gemma_textvqa_val5000_hygon_before'
base=object(); model=EngineModel(ADAPTER,base)
state={f'block{i}.lora_{letter}.weight':2*i+j for i in range(238) for j,letter in enumerate('AB')}
eq=lambda a,b:a==b
proof={}
with native.observe_lora(model,base,ADAPTER,state,eq,proof) as called:
    assert model.layers['block0'].forward(adapter_names=[ADAPTER])=='fixture LORA compute'
    assert called=={'block0'}
rejected(c.compare_layer_execution,model.layers['block0'],ADAPTER,{'adapter_names':['__base__']})
rejected(c.compare_layer_execution,model.layers['block0'],ADAPTER,{'adapter_names':['wrong']})
model.layers['block0'].disable_adapters=True
rejected(c.compare_layer_execution,model.layers['block0'],ADAPTER,{'adapter_names':[ADAPTER]})
model.layers['block0'].disable_adapters=False
model.params['block0.lora_A.'+ADAPTER+'.weight']._version=1
try:
    with native.observe_lora(model,base,ADAPTER,state,eq,proof):pass
except ValueError:pass
else:raise AssertionError('live changed state accepted')

now=100
rt={'vendor':'fixture'}
permit=dict(request='a'*32,arm=ARM,checkpoint_SHA256=c.CP[ARM],vendor='HYGON_HIP',runtime=rt,
            identity_SHA256=c.IDENTITIES,question_projection_SHA256='1'*64,media_readback_SHA256='2'*64,
            root_single_native_arm=True,independent_current_runtime=True,gold_allowed=False,
            authorized_start_epoch=99,deadline_epoch=101,media_bytes_and_64_train_disjoint_verified=True)
native.owner_preflight(permit,ARM,rt,c.IDENTITIES,'1'*64,'2'*64,now)
rejected(native.owner_preflight,dict(permit,checkpoint_SHA256=c.CP['Hygon']),ARM,rt,c.IDENTITIES,'1'*64,'2'*64,now)
rejected(native.owner_preflight,dict(permit,gold_allowed=True),ARM,rt,c.IDENTITIES,'1'*64,'2'*64,now)


def exercise(active):
    def generate(**_):
        return model.layers['block0'].forward(adapter_names=[ADAPTER] if active else ['__base__'])
    tpl=NS(template_meta=NS(template_type='gemma3_vision'),generate=generate)
    def infer(req,config,**kwargs):
        assert kwargs['adapter_request'].name==ADAPTER and config.max_tokens==32
        assert req[0].messages[0]['role']=='user' and req[0].messages[0]['content'].endswith('using a short phrase.')
        tpl.generate(generation_config=NS(do_sample=False,num_beams=1,max_new_tokens=32),adapter_names=[ADAPTER])
        return [NS(choices=[NS(finish_reason='stop',message=NS(content='fixture answer'))])]
    attempts=0
    def stop():
        nonlocal attempts
        attempts+=1
        if attempts>1:raise TimeoutError
    summary={'activated_requests':0}
    swift=NS(AdapterRequest=NS,RequestConfig=NS,InferRequest=NS)
    with patch.object(native,'file_stamp',return_value=None):
        entries,fallback=native.execute_native(NS(model=model,infer=infer),tpl,base,swift,NS(equal=eq),
                    ARM,HERE,state,rows,HERE,stop,summary)
    return entries,fallback,summary

ok,tail,summary=exercise(True)
assert len(ok)==1 and ok[0]['status']=='scored' and tail=='DEADLINE' and summary['activated_requests']==1
failed,tail,memo=exercise(False)
assert failed[0]['reason']==tail=='ACTIVATION_NOT_PROVED' and memo['activated_requests']==0
with TemporaryDirectory() as tmp:
    path=Path(tmp)/'predictions-private.jsonl'
    rec=native.seal_arm(path,roster,failed,tail,ARM,memo)
    assert rec['planned']==rec['failed']==5000 and rec['scored']==0
    scorer=runpy.run_path(str(REPO/'tools/textvqa_effect.py'))
    scored=scorer['score_arm'](path.read_bytes(),str(path),[str(i) for i in roster],{},None,'raw')
    assert len(scored)==5000 and all(x['textvqa_credit_x30']==0 for x in scored)
    all_arms={arm:dict(path=arm+'.jsonl',SHA256='3'*64,checkpoint_SHA256=c.CP[arm]) for arm in c.ARMS}
    assert c.scorer_manifest('9'*64,all_arms)['arms']==all_arms
    all_arms[ARM]['checkpoint_SHA256']=c.CP['Hygon']
    rejected(c.scorer_manifest,'9'*64,all_arms)
print(json.dumps(dict(passed=True,actual_models=0,actual_GPU=0,actual_gold_reads=0,fixture_questions=5000,
                     fixture_images=3166,fixture_native_requests=2,planned_across_three=15000,formal_READY=False),sort_keys=True))
