"""One new tuple-contract counterexample check, no old frame fixture or real artifact."""
import ast,copy,hashlib,json,math,re
from pathlib import Path
D=Path(__file__).resolve().parent
G=D.parents[3]
O=G/'local-only/GLM474-original177-corrected-tuple-new-boundary-check-v1'
def main():
    O.mkdir();source=(D/'attention_evidence.py').read_bytes();tree=ast.parse(source)
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','validate_tuple')]
    ns={};exec(compile(ast.Module(body=body,type_ignores=[]),str(D/'attention_evidence.py'),'exec'),ns)
    base=dict(site='projection',m=1313,n=1792,k=512,row=0,column=0,stride=512,frame=dict(tokens=1,row_past_count=[[0,1312,1]]),global_rank=0,stack=[('history.selected_value',{})],branches=['selection.complete_prefix','v398.enabled','history.dispatch.standard'])
    checks=[]
    def test(name,fn,reject=False):
        try:fn()
        except (ValueError,AssertionError,TypeError) as e:checks.append(dict(name=name,expected='reject' if reject else 'accept',passed=reject,error_type=type(e).__name__))
        else:checks.append(dict(name=name,expected='reject' if reject else 'accept',passed=not reject))
    def call(**changes):return ns['validate_tuple'](**dict(base,**changes))
    for past in (1312,1313,1314):test('actual_owner0_history_past%d'%past,lambda past=past:call(m=past+1,frame=dict(tokens=1,row_past_count=[[0,past,1]])))
    test('rank8_uses_global_row_offset',lambda:call(global_rank=8,row=8*1792))
    test('nonhistory_projection_keepsM1',lambda:call(m=1,n=2048,k=6144,stride=6144,stack=[]))
    test('output_keepsM1',lambda:call(site='output',m=1,n=6144,k=1024,stride=1024,stack=[],column=8192))
    bad=[('history_wrong_M',dict(m=1)),('history_wrong_past',dict(m=1314)),('history_wrong_N',dict(n=1791)),('history_wrong_K',dict(k=1024,stride=1024)),('wrong_stride',dict(stride=511)),('rank8_cannot_reuse_rank0offset',dict(global_rank=8,row=0)),('wrong_column',dict(column=1)),('outside_history_bulk_M',dict(stack=[])),('fresh_window_bulk_M',dict(stack=[('fresh.expand_cached',{})])),('missing_complete_prefix',dict(branches=['v398.enabled','history.dispatch.standard'])),('partial_prefix',dict(branches=['selection.partial_prefix','v398.enabled','history.dispatch.standard'])),('packed_history',dict(branches=['selection.complete_prefix','v398.enabled','history.dispatch.packed'])),('v398_disabled',dict(branches=['selection.complete_prefix','v398.disabled','history.dispatch.standard'])),('not_trueT1',dict(frame=dict(tokens=2,row_past_count=[[0,1312,2]]))),('multiblock_future_prefix',dict(m=8193,frame=dict(tokens=1,row_past_count=[[0,8192,1]]))),('foreign_global_rank',dict(global_rank=16)),('zero_M',dict(m=0))]
    for name,changes in bad:test(name,lambda changes=changes:call(**changes),True)
    def roundtrip():
        raw_values=[]
        for name,target,token in [('reader.py','body','report'),('launch_once.py','body',"v['report']")]:
            t=ast.parse((D/name).read_bytes());expressions=[n.value for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==target for x in n.targets) and 'json.dumps' in ast.unparse(n.value) and token in ast.unparse(n.value)]
            assert len(expressions)==1
            value={'corrected':False,'label':'中文'};raw=eval(compile(ast.Expression(expressions[0]),name,'eval'),dict(json=json,report=value,v={'report':value}));assert raw.endswith(bytes((10,))) and json.loads(raw)==value;raw_values.append(raw)
        assert raw_values[0]==raw_values[1]
    test('real_report_serialization_roundtrip',roundtrip)
    result=dict(passed=all(c['passed'] for c in checks),source_SHA256=hashlib.sha256(source).hexdigest(),normal=sum(c['expected']=='accept' for c in checks),reject=sum(c['expected']=='reject' for c in checks),checks=checks,failures=[c for c in checks if not c['passed']],old_fixtures_run=0,real_owner_reads=0,remote_calls=0,Model_calls=0)
    raw=(json.dumps(result,indent=2)+chr(10)).encode();(O/'result.json').write_bytes(raw);print(json.dumps(dict(passed=result['passed'],normal=result['normal'],reject=result['reject'],failures=result['failures'],SHA256=hashlib.sha256(raw).hexdigest())))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
