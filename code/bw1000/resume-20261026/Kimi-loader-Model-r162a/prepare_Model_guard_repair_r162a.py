"""Local immutable namespace repair. No SSH, Slurm, Git, Runtime, or dlopen."""
from pathlib import Path
import ast, hashlib, json, shutil, datetime

V=Path(__file__).resolve().parent
OLD=V/'Kimi-loader-Model-private-r162'
D=V/'Kimi-loader-Model-private-r162a'
assert not D.exists()
shutil.copytree(OLD,D)
for p in sorted(D.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if p.is_file():
        raw=p.read_bytes()
        if p.suffix in ('.py','.json','.sh','.md'):p.write_bytes(raw.replace(b'r162',b'r162a'))
    if 'r162' in p.name:p.rename(p.with_name(p.name.replace('r162','r162a')))
M=D/'model'
for p in [D/'prepare_Model_r162a.py',D/'refresh_Model_pins_r162a.py',D/'source-manifest-r162a.json',D/'CPU-model-contract-observed-r162a.json']:
    if p.exists():p.unlink()

contract='''"""Fixed six-case pilot queue validation; CPU-only and no runtime imports."""
import json
def validate_frozen_pilot(model_directory):
    proof=json.loads((model_directory/'fixed-data-r162a.json').read_bytes())
    expected={'case_count':6,'C1_cases':4,'C8_cases':2,'request_executions':20,'unique_question_ids':16,'C1_is_subset_of_C8':True,'not_new32_heldout':True,'cap':512,'context':1024,'W':8,'strategy':'state','W2pair_fixed_ON':True,'W13_fixed_OFF':True,'registry_cache_fixed_ON':True,'model_profile_OFF':True,'EOS_not_forced':True,'root_only_GPU_GO':True}
    assert proof==expected,'fixed pilot metadata changed'
    commands=[json.loads(p.read_bytes())for p in sorted((model_directory/'queue').glob('q*.json'))]
    assert len(commands)==7 and commands[-1]['action']=='stop'
    cases=commands[:-1]
    assert all(c['action']=='case'for c in cases)
    assert [c['C']for c in cases]==[1,1,1,1,8,8]
    assert len({c['id']for c in cases})==6
    for c in cases:
        assert len(c['requests'])==c['C']
        assert c['W']==8 and c['strategy']=='state'and c['cap']==512 and c['W2pair']and not c['model_profile']
        assert not any('W13'in k or 'partial'in k for k in c)
    c1={q['id']for c in cases[:4]for q in c['requests']}
    c8={q['id']for c in cases[4:]for q in c['requests']}
    assert len(c1)==4 and len(c8)==16 and c1<=c8
    assert sum(len(c['requests'])for c in cases)==20
    return {'passed':True,'case_count':6,'request_executions':20,'unique_question_ids':16,'C1_is_subset_of_C8':True}
'''
(M/'launcher/pilot_queue_contract.py').write_bytes(contract.encode())
p=M/'launcher/model_keeper.py';s=p.read_text()
old="proof=json.loads((src.parent/'fixed-data-r162a.json').read_bytes());assert proof['GSM_unique']==32 and proof['case_count']==12 and proof['no_new_tokenization']"
assert s.count(old)==1
s=s.replace(old,"from pilot_queue_contract import validate_frozen_pilot\n w['frozen_pilot_contract']=validate_frozen_pilot(src.parent)")
p.write_bytes(s.encode())
# Test exactly the keeper's queue contract, including deliberate stale metadata rejection.
p=D/'CPU_model_contract_r162a.py';s=p.read_text()
s=s.replace("tests=[];q=", "import sys\nsys.path.insert(0,str(M/'launcher'))\nfrom pilot_queue_contract import validate_frozen_pilot\ntests=[];assert validate_frozen_pilot(M)['passed'];tests.append('keeper_actual_frozen_pilot_contract_executed');q=")
p.write_bytes(s.encode())

context=M/'launcher/run-context-public.json';c=json.loads(context.read_bytes())
c['run']='kimi-loader-current-r162a-a7260d91';context.write_bytes((json.dumps(c,indent=2)+'\n').encode())
# Same reviewed code, namespace only, with exact queue validation added before any scheduling.
review=D/'MODEL-REVIEW.zh-CN.md'
review.write_bytes(('''# r162a：修复必要加载的提交前队列校验

旧 r162 的实际 CPU keeper 910407.515 在 validate 阶段以 KeyError GSM_unique 失败，GPU_submitted=false，groups=[]，没有开始模型加载。旧结果和 namespace 保留。

本包只将所有运行/结果/源码路径改为独立 r162a，并将陈旧的 GSM32/12case/no_new_tokenization guard 改为实际 frozen queue 校验。keeper 与 CPU source contract 共用 pilot_queue_contract.py：6case，C1四题+C8两组，20次执行、16独立题、C1subset，原 cap512/context1024/W8/state/W2pairON/cacheON/W13OFF。CPU source contract 仍逐题与原冻结题目/答案比较，源码SHA冻结后由 keeper 验证。未删除生命周期/网络/内存/capacity/质量条件。

复用 CPU511 Native798c4bca01cddb0755a91171ef61dfa755329518bb8cfc87640305a24e55a6e1，不重新编译、不添加GPU优化。固定ON加载只能归因当前加载工作，不能声称旧冷加载的因果加速。没有SSH/Git/Slurm，没有实际GO。实际部署目录 /refactor/incoming/Kimi-sparse-profile-Model-r162a/，入口 launcher/preflight_then_run.py，root审阅发布后生成对应GO并执行。
''').encode())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prefix='/refactor/incoming/Kimi-sparse-profile-Model-r162a/'
mp=M/'runtime/model-manifest-r9183.json';manifest=json.loads(mp.read_bytes())
for path in manifest['source_SHA256']:
    if path.startswith(prefix):manifest['source_SHA256'][path]=sha(M/path[len(prefix):])
mp.write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
gp=M/'launcher/GPU-GO-schema.json';g=json.loads(gp.read_bytes());g['manifest_SHA256']=sha(mp);gp.write_bytes((json.dumps(g,indent=2)+'\n').encode())
pin=M/'launcher/model-source-pins.json';pins=json.loads(pin.read_bytes());pins['manifest_SHA256']=sha(mp)
producer=M/'producer-source-r162a.json'
pins['files']={prefix+f.relative_to(M).as_posix():sha(f)for f in M.rglob('*')if f.is_file()and f not in (pin,producer)}
pin.write_bytes((json.dumps(pins,indent=2)+'\n').encode())
prod=json.loads(producer.read_bytes());prod['manifest_SHA256']=sha(mp)
prod['repair']={'old_failed_CPU_step':'910407.515','old_GPU_submitted':False,'old_error':'KeyError GSM_unique','fixed':'shared actual six-case queue validation before scheduling','Native_recompiled':False}
prod['files_SHA256']={f.relative_to(M).as_posix():sha(f)for f in M.rglob('*')if f.is_file()and f!=producer}
producer.write_bytes((json.dumps(prod,indent=2)+'\n').encode())
for p in M.rglob('*.py'):ast.parse(p.read_text())
assert all(b'\r\n'not in p.read_bytes()for p in M.rglob('*.sh'))
print(json.dumps({'prepared':True,'directory':str(D),'Model_manifest_SHA256':sha(mp),'source_pins_SHA256':sha(pin),'producer_SHA256':sha(producer),'GPU':False}))
