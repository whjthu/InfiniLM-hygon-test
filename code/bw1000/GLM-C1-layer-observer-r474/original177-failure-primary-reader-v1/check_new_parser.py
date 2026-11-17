"""One new synthetic parser/clock/serialization check; never real original artifacts."""
import ast,copy,hashlib,json,math,os,re,stat,types
from pathlib import Path
D=Path(__file__).resolve().parent
G=D.parents[3]
O=G/'local-only/GLM474-original177-primary-reader-new-parser-check-v1'

def main():
    O.mkdir();source=(D/'reader.py').read_bytes();tree=ast.parse(source)
    names={'need','strict','identity','ishash','real','safe_exception','validate_slot'}
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(ast=ast,hashlib=hashlib,json=json,math=math,Path=Path,re=re,H=lambda b:hashlib.sha256(b).hexdigest())
    exec(compile(ast.Module(body=body,type_ignores=[]),str(D/'reader.py'),'exec'),ns)
    rows=[]
    def test(name,call,reject=False):
        try:call()
        except (ValueError,TypeError,AssertionError) as e:
            rows.append(dict(name=name,expected='reject' if reject else 'accept',passed=reject,error_type=type(e).__name__))
        else:rows.append(dict(name=name,expected='reject' if reject else 'accept',passed=not reject))
    pin=dict(ordinal=1,case_id='synthetic',suite='synthetic',prompt_tokens=1,prompt_IDs_SHA256='a'*64,body_SHA256='b'*64)
    def row(ids):
        ticks=[(i+1)*10**9 for i in range(len(ids))];visible=[t for token,t in zip(ids,ticks) if token not in {2,7}]
        return dict(pin,diagnostic=True,screen_index=None,completed=True,reset_before=True,reset_after=True,generated_token_ids=ids,generated_token_ids_SHA256=ns['H'](json.dumps(ids,separators=(',',':')).encode()),planned_max_new_tokens=4096,natural_EOS=True,cap_hit=False,sampled_ns=ticks,prefill_fragment_plan=[dict(offset=0,tokens=1,IDs_SHA256='c'*64)],runtime_prefill_chunk=512,original_request_prefill_window=4096,prefill_CAPI_s=[.1],decode_CAPI_s=[.1]*(len(ids)-1),Native_forward_begin=0,Native_forward_end=len(ids),whole_backend_s=10.,request_enter_ns=500000000,visible_nonEOS_nonspecial=len(visible),backend_decode_tok_s=(len(visible)-1)*1e9/(visible[-1]-visible[0]) if len(visible)>1 else None,backend_sampled_TTFT_s=(visible[0]-500000000)/1e9 if visible else None)
    check=lambda r:ns['validate_slot'](r,0,[pin],{2},{7})
    for name,ids,rate,ttft in [('zero_visible',[2],None,None),('one_visible',[10,2],None,.5),('two_visible',[10,11,2],1.,.5),('special_filtered',[7,10,11,2],1.,1.5)]:
        def valid(ids=ids,rate=rate,ttft=ttft):
            v=check(row(ids));assert v['backend_decode_tok_s']==rate and v['backend_sampled_TTFT_s']==ttft
        test(name,valid)
    base=row([10,11,2])
    for name,k,v in [('wrong_rate','backend_decode_tok_s',2.),('wrong_ttft','backend_sampled_TTFT_s',.7),('backwards_ticks','sampled_ns',[2000000000,1000000000,3000000000]),('future_enter','request_enter_ns',2000000000),('wrong_output_hash','generated_token_ids_SHA256','f'*64),('missing_completion','completed',False),('foreign_input','prompt_IDs_SHA256','f'*64)]:
        test(name,lambda k=k,v=v:check(dict(base,**{k:v})),True)
    test('strict_finite_json',lambda:ns['strict']('{"x":1.5}'))
    for name,raw in [('duplicate_json','{"x":1,"x":2}'),('json_nan','{"x":NaN}'),('json_overflow','{"x":1e999}')]:test(name,lambda raw=raw:ns['strict'](raw),True)
    test('safe_unknown_exception',lambda: (_ for _ in ()).throw(AssertionError()) if ns['safe_exception']("ValueError('private endpoint')",'',{'known'})['known_constant_message'] is not None else None)
    def roundtrip():
        expressions=[]
        for filename,target,token in [('reader.py','raw','report'),('launch_once.py','body',"v['report']")]:
            t=ast.parse((D/filename).read_bytes())
            options=[n.value for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==target for x in n.targets) and token in ast.unparse(n.value) and 'json.dumps' in ast.unparse(n.value)]
            assert len(options)==1
            value={'中文':'边界','passed':False};raw=eval(compile(ast.Expression(options[0]),filename,'eval'),{'json':json,'report':value,'v':{'report':value}})
            assert raw.endswith(b'\n') and not raw.endswith(bytes((92,110))) and json.loads(raw)==value
            expressions.append(raw)
        assert expressions[0]==expressions[1]
    test('actual_source_serialization_roundtrip',roundtrip)
    passed=all(r['passed'] for r in rows)
    result=dict(passed=passed,source_SHA256=hashlib.sha256(source).hexdigest(),normal=sum(r['expected']=='accept' for r in rows),reject=sum(r['expected']=='reject' for r in rows),checks=rows,all_failures=[r for r in rows if not r['passed']],original_artifact_reads=0,remote_calls=0,Torch_imports=0,old_tests=0)
    raw=(json.dumps(result,indent=2)+'\n').encode();(O/'result.json').write_bytes(raw);print(json.dumps(dict(passed=passed,normal=result['normal'],reject=result['reject'],result_SHA256=hashlib.sha256(raw).hexdigest(),failures=result['all_failures'])))
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
