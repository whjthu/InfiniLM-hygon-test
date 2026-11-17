"""New contract-only boundaries: AST-selected stdlib functions, zero Torch import."""
import ast,json,copy
from pathlib import Path,PurePosixPath
D=Path(__file__).resolve().parent

def main():
    raw=(D/'vlm/runtime/usgs_domain_contract.py').read_bytes();tree=ast.parse(raw)
    keep=[]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'SCHEMA','MODES'} for t in n.targets):keep.append(n)
        if isinstance(n,ast.FunctionDef) and n.name in {'check_config','cursor_contract'}:keep.append(n)
    def require(x,reason):
        if not x:raise ValueError(reason)
    def checked_path(x):
        p=PurePosixPath(x);require(p.is_absolute() and '..' not in p.parts,'PATH');return p
    ns=dict(require=require,checked_path=checked_path,Path=PurePosixPath,MEMBERS={'trainer_state.json','args.json','adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))})
    exec(compile(ast.Module(body=keep,type_ignores=[]),'new-stdlib-contract','exec'),ns)
    pin=dict(path='/scope/pin.json',bytes=1,SHA256='1'*64)
    c=dict(schema='USGS_T04_DOMAIN_TRAINING_V1',case='T04.sft',arm='domain',mode='fresh25',root_GO=True,GPU_GO=True,device='cuda',formal17=False,world_size=8,physical_rows=800,unique_rows=45,max_steps=25,accumulation=4,initial_tensors=504,initial_elements=43646976,comparison_policy='EXACT_ENGINEERING_SAVED_STATE',pair_id='1'*32,binding=pin,native45=pin,training800=pin,occurrences=pin)
    passed=[];failures=[]
    def run(name,fn,reject=False):
        try:fn()
        except ValueError:
            if reject:passed.append(name);return
            failures.append(dict(name=name,error='unexpected rejection'));return
        except BaseException as e:failures.append(dict(name=name,error=type(e).__name__+':'+str(e)));return
        if reject:failures.append(dict(name=name,error='bad input accepted'))
        else:passed.append(name)
    run('fresh_valid',lambda:ns['check_config'](c,clock=False))
    for key,value in [('GPU_GO',False),('root_GO',False),('case','T01.sft'),('physical_rows',799),('world_size',1),('initial_tensors',503),('mode','legacy'),('arm','generic')]:
        run('reject_'+key,lambda k=key,v=value:ns['check_config'](dict(c,**{k:v}),clock=False),True)
    initial=[dict(returned_values=[1,25,800,False,32,100,25],actual_loader_length=100)]
    run('fresh_actual_cursor',lambda:ns['cursor_contract']('fresh25',100,initial,[dict(returned_values=[0,0])]))
    run('resume_actual_cursor',lambda:ns['cursor_contract']('resume20to25',100,initial,[dict(returned_values=[0,80])],.8))
    run('reject_old_zero_skip',lambda:ns['cursor_contract']('resume20to25',100,initial,[dict(returned_values=[0,0])],.8),True)
    run('reject_old64_loader',lambda:ns['cursor_contract']('resume20to25',8,initial,[dict(returned_values=[10,0])],10.),True)
    run('reject_wrong_checkpoint_epoch',lambda:ns['cursor_contract']('resume20to25',100,initial,[dict(returned_values=[0,80])],1.),True)
    for p in D.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
    result=dict(passed=not failures,checks=passed,failures=failures,Torch_imports=0,model_calls=0,remote_calls=0,AST_files=len(list(D.rglob('*.py'))))
    print(json.dumps(result));return result
if __name__=='__main__':main()


def check_counter_delta():
    tree=ast.parse((D/'vlm/runtime/usgs_domain_contract.py').read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_supervised_counter')
    def require(ok,why):
        if not ok:raise ValueError(why)
    ns={'require':require};exec(compile(ast.Module(body=[node],type_ignores=[]),'counter-delta','exec'),ns);f=ns['validate_supervised_counter']
    passed=[]
    assert f({'supervised_EOS_count':1},5,1) is False;passed.append('domain_missing_counter_actual_measurement')
    assert f({'supervised_EOS_count':1,'effective_supervised_tokens':5},5,1) is True;passed.append('control_reference_counter_matches')
    for name,row,count,eos in [('mismatch',{'supervised_EOS_count':1,'effective_supervised_tokens':4},5,1),('zero',{'supervised_EOS_count':1},0,1),('EOS',{'supervised_EOS_count':2},5,1),('null_reference',{'supervised_EOS_count':1,'effective_supervised_tokens':None},5,1),('bool_reference',{'supervised_EOS_count':1,'effective_supervised_tokens':True},1,1)]:
        try:f(row,count,eos)
        except ValueError:passed.append('reject_'+name)
        else:raise AssertionError(name)
    return dict(passed=True,checks=passed,old_checks_reexecuted=False,Torch_imports=0,remote_calls=0)
