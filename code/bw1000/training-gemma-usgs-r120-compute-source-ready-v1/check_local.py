"""Offline full carrier, same-step argv and real bounded file-visibility regressions."""

from hashlib import sha256
import ast
from base64 import b64decode
import json
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory
from threading import Timer
import time
from types import SimpleNamespace
from unittest.mock import patch
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import bootstrap as b
import prepare as p


def check():
    ctx=p.carrier()
    a=ctx["a"]
    stable,schema,manifest=ctx["verified_stable"]()
    claim=json.loads(stable["CLAIM.json"])
    claim.update(root_GO=True,GPU_GO=True,training_GO=True)
    members = stable | {"CLAIM.json":a.encode(claim)}
    effective=ctx["source_manifest"](members,schema)
    expected=b.pin(effective)
    target=b.contract_projection(members["contract.py"])
    del target["ROOT"]
    def status():
        return dict(rounds=0,enoent_rounds=0,checked_last_round=0,last_member=None,errno=None)
    def rejects(fn,code):
        try: fn()
        except b.Bad as exc: assert str(exc)==code, str(exc)
        else: raise AssertionError("expected failed visibility")
    with TemporaryDirectory(dir=HERE) as dirname:
        root=Path(dirname)
        for name,data in members.items():
            if name!="verify_original.py": (root/name).write_bytes(data)
        (root/"SOURCE_FILES.json").write_bytes(effective)
        record=status()
        pending=root/"delayed-complete-file"
        pending.write_bytes(members["verify_original.py"])
        timer=Timer(.20,lambda: pending.replace(root/"verify_original.py"))
        timer.start()
        try:
            now=time.time()
            b.wait_ready(root,expected,target,now-1,now+58,record)
        finally:
            timer.join()
        assert record["rounds"]>=2 and record["enoent_rounds"]>=1 and record["checked_last_round"]==32
        body=members["verify_original.py"]
        (root/"verify_original.py").write_bytes(b" "+body[1:])
        record=status()
        now=time.time()
        rejects(lambda:b.wait_ready(root,expected,target,now-1,now+58,record),"MEMBER_SHA256_MISMATCH")
        assert record["rounds"]==1 and record["enoent_rounds"]==0
        (root/"verify_original.py").unlink()
        with patch.object(b,"MAX_WAIT",.45):
            record=status()
            now=time.time()
            rejects(lambda:b.wait_ready(root,expected,target,now-1,now+58,record),"SOURCE_VISIBILITY_TIMEOUT")
            assert 1<=record["enoent_rounds"]<=record["rounds"]
        rejects(lambda:b.budget(100,30,160),"NO_BUDGET_FOR_PRE_AND_ORIGINAL_SECOND_ADMISSION")
        rejects(lambda:b.budget(100,95,161),"PRE_STEP_EPOCH_BOUNDS")
        assert b.budget(100,55,155)==20 and b.budget(100,31,160)==1

    # Extract the actual generated function rather than restating its argv.
    controller=ast.parse(members["controller.py"])
    function,=[n for n in controller.body if isinstance(n,ast.FunctionDef) and n.name=="compute_source_command"]
    root=PurePosixPath(target["R"])
    captured=dict(time=SimpleNamespace(time=lambda:110), D=root/"results"/p.LABEL/"src",R=root,
                  LABEL=p.LABEL,PARENT=a.PARENT,NODE=a.NODE,REQUEST=p.SCOPE,DISPATCH_REQUEST=p.DISPATCH,
                  HOST_PYTHON="/usr/local/python3.12/bin/python3",json=json,need=b.need,verify=lambda _:expected,
                  stable_json=lambda path,cap: (
                      (dict(placement_authority_SHA256="a"*64),None) if path.name=="CLAIM.json" else
                      (dict(request_id=p.SCOPE,issued_epoch=100),dict(SHA256="a"*64))))
    fake_contract=SimpleNamespace(PORT=p.PORT,TRAINING_REQUEST=p.SCOPE)
    with patch.dict(sys.modules,contract=fake_contract):
        exec(compile(ast.Module(body=[function],type_ignores=[]),"controller.py","exec"),captured)
        argv=captured["compute_source_command"](None,60)
    assert argv[:5]==["/usr/local/python3.12/bin/python3","-I","-S","-B","-c"]
    assert argv[5].encode()==(HERE/"bootstrap.py").read_bytes()
    assert json.loads(argv[6])==dict(root=str(captured["D"]), manifest=expected,target=target,issued=100,pre_deadline=170)
    assert "os.execv(sys.executable" in argv[5]

    # Use the prior real capsule fixture with the new production carrier and scope.
    old=p.pinned(p.PRIOR/"check_local.py","3ce8ffd403889fa8e0948a5a6ea8a29302fdc70ad1e3df0b820d9b3e13c92af7").decode()
    old=p.splice(old,"import admission as a\nimport prepare as p","p=new_carrier\na=p.a")

    def prepare_checked(output,*args):
        result=ctx["prepare"](output,*args)
        wire=json.loads((output/"WIRE_DATA.json").read_bytes())
        body=zlib.decompress(b64decode(wire["body"],validate=True))
        assert len(body)<=280000 and sha256(body).hexdigest()==wire["body_SHA256"]
        source=body.decode()
        # Execute the exact receiver locally after replacing only its target path
        # and replacing process exec with a capture to avoid running the owner.
        place=output/"local-receiver"
        identity,=[line for line in source.splitlines() if line.startswith("remote_root=")]
        source=p.splice(source,identity,"remote_root="+repr(str(place)))
        root_check,=[line for line in source.splitlines() if line.startswith("assert remote_root.startswith")]
        source=p.splice(source,root_check,"assert remote_root=="+repr(str(place)))
        calls=[]
        with patch("os.execv",side_effect=lambda *argv:calls.append(argv)):
            exec(compile(source,"<fixture exact compact receiver>","exec"),{})
        effective=output/"effective-source"
        assert len(calls)==1 and sorted(f.name for f in (place/"src").iterdir())==sorted(f.name for f in effective.iterdir())
        assert all(f.read_bytes()==(place/"src"/f.name).read_bytes() for f in effective.iterdir())
        return result

    exec(compile(old,"<prior carrier fixture using new scope>","exec"),
         dict(__name__="__main__",__file__=str(__file__),
              new_carrier=SimpleNamespace(**(ctx | {"prepare":prepare_checked}))))
    print("PASS delayed final actual file; corrupted hash no retry; timeout; reserved authority boundary; inline pre argv")


if __name__=="__main__":
    check()
