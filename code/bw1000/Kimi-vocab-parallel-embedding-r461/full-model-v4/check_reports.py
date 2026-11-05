"""Only changed actual producer/consumer file glue, original four sealed reports.

Function bodies come from production AST, not a second implementation. No SDK.
"""
from pathlib import Path
import ast,hashlib,json,tarfile,tempfile,time,types

def function(path,name,environment):
    source=ast.parse(path.read_bytes())
    fn,=[n for n in ast.walk(source)if isinstance(n,ast.FunctionDef)and n.name==name]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),environment)
    return environment[name]

def main(archive):
    d=Path(__file__).resolve().parent
    assert hashlib.sha256(Path(archive).read_bytes()).hexdigest()=='8f9eb998ba7a666639ad3b0dd9fbf8d8226c0d5987f1dbd151ed1cd2a16e1192'
    with tarfile.open(archive)as tar:
        raw={i:tar.extractfile('kimi/node%d.json'%i).read()for i in range(4)}
    original={i:json.loads(b)for i,b in raw.items()}
    q=original[0]
    with tempfile.TemporaryDirectory()as tmp:
        folder=Path(tmp);producer=dict(json=json)
        save=function(d/'node.py','save',producer)
        def publish(i,value):
            producer.update(report=folder/('node%d.json'%i),result=value);save()
        for i,v in original.items():publish(i,v)
        environment=dict(folder=folder,RUN=q['run'],a=types.SimpleNamespace(manifest_sha=q['source_manifest_SHA256']),epoch=q['epoch'],expected={},NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12'],json=json,hashlib=hashlib,time=time,w={})
        reports=function(d/'controller.py','reports',environment)
        assert reports()==[]and len(environment['expected'])==4 # Original reports are genuinely incomplete.
        actual_read=Path.read_bytes;injected=[]
        def read(path):
            if path==folder/'node1.json'and not injected:
                assert path.exists();injected.append(1);raise FileNotFoundError(2,'recorded ENOENT after real exists',str(path))
            return actual_read(path)
        Path.read_bytes=read
        try:assert reports()==[]
        finally:Path.read_bytes=actual_read
        assert injected==[1]and environment['w']['report_visibility']['1']['misses']==1
        assert reports()==[] # Same file subsequently reads, no fake completion.
        (folder/'node2.json').unlink();assert reports()==[]
        publish(2,original[2]);assert reports()==[]and not(folder/'node2.tmp').exists()
        publish(0,dict(original[0],source_manifest_SHA256='0'*64))
        try:reports()
        except AssertionError:pass
        else:raise AssertionError('Identity mismatch was swallowed')
        publish(0,original[0]);publish(3,dict(original[3],error='recorded owner lifecycle failure'))
        try:reports()
        except RuntimeError:pass
        else:raise AssertionError('Actual owner error was swallowed')
        publish(3,original[3])
        # Execute the unchanged terminal visibility block with only its clock/sleep leaf recorded.
        tree=ast.parse((d/'controller.py').read_bytes());main,=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main'];body=next(n.body for n in main.body if isinstance(n,ast.Try))
        start=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='visibility_limit'for t in n.targets))
        block=body[start:start+4];clock=types.SimpleNamespace(now=0.)
        environment.update(cleanup_limit=1000.,time=types.SimpleNamespace(monotonic=lambda:clock.now,sleep=lambda seconds:setattr(clock,'now',clock.now+seconds)))
        try:exec(compile(ast.Module(body=block,type_ignores=[]),'actual-terminal30s','exec'),environment)
        except AssertionError:pass
        else:raise AssertionError('Incomplete original reports were accepted at terminal')
        assert clock.now==30.
        # One explicitly recording completion leaf exercises the new success pin.
        environment['time']=time
        publish(0,dict(original[0],completed=True))
        completed=reports();assert len(completed)==1
        path,value,pin=completed[0];raw=path.read_bytes()
        assert pin['SHA256']==hashlib.sha256(raw).hexdigest()and pin['bytes']==len(raw)
        path.unlink()
        assignment=next(n for n in body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='w'and isinstance(t.slice,ast.Constant)and t.slice.value=='node_reports'for t in n.targets))
        environment['values']=completed
        Path.read_bytes=lambda p:(_ for _ in()).throw(AssertionError('terminal pin reopened a report'))
        try:exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-final-snapshot-pins','exec'),environment)
        finally:Path.read_bytes=actual_read
        assert environment['w']['node_reports']==[pin]
    return dict(passed=True,GPU=0,SDK=False,new_jobs=0,actual_producer_atomic_save=True,actual_reports_body=True,original_four_reports=True,ENOENT_once_then_recovered=True,atomic_republish=True,identity_and_owner_error_hard_rejected=True,original_terminal30s_incomplete_rejected=True,original_reports_not_reclassified_completed=True,completed_leaf_explicitly_recording=True,same_parsed_bytes_SHA_length=True,actual_terminal_assignment_no_reopen=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args();print(json.dumps(main(a.archive),indent=2))
