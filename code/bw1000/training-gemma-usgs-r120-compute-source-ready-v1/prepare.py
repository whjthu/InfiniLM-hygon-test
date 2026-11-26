"""Rebind the prior audited one-shot R120 carrier with a same-step visibility gate."""

import argparse
import ast
from base64 import b64decode, b64encode
from hashlib import sha256
import json
from pathlib import Path
import sys
import types
import zlib


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "training-gemma-usgs-r120-runtime-admission-v1"
OLD_LABEL = PRIOR.name
OLD_SCOPE = "568be365fb2444a8bf0c4a30bb2b8d13"
OLD_DISPATCH = "67fd59f0a8574fdaab24324c5dea04cf"
LABEL = HERE.name
SCOPE = "886fb0896c094de9acc384dc0d13ebb2"
DISPATCH = "b298f9ce6be74ce6b9035fc2f61d7f2e"
PORT = 32129
PREVIOUS_FAILURE = "87703aeb01c7945458938639a531fc4442ecb847d5793d96818e2f565df631fd"
ADMISSION_SHA = "465f80f86c2c81818beb8e5578cf53622e1958b71ec0b6e2ac140bdb86a82c0f"
CARRIER_SHA = "c5e4e4d8c35e935927968ae68079b931856614b4c04b32f747a28ff0cfa8b02c"
PRIOR_MANIFEST = "bc6124149791a510fe7f4dca7adaca142eab5be8527e0bfb52677da34d03250b"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(path,digest):
    raw=path.read_bytes()
    require(0<len(raw)<=65536 and sha256(raw).hexdigest()==digest,"EXACT_PREVIOUS_SOURCE_PIN")
    return raw


def splice(text,anchor,value):
    require(text.count(anchor)==1,"EXACT_SINGLE_SOURCE_ANCHOR")
    return text.replace(anchor,value,1)


def moved(raw):
    for before,after in ((OLD_LABEL,LABEL),(OLD_SCOPE,SCOPE),(OLD_DISPATCH,DISPATCH),("32128",str(PORT)),
                         ("/tmp/gr120a","/tmp/gr120b")):
        raw=raw.replace(before.encode(),after.encode())
    return raw


def gated_controller(raw,bootstrap):
    inline_blob=b64encode(zlib.compress(bootstrap,9))
    hook = (
        "def compute_source_command(clock, seconds):\n"
        "    from base64 import b64decode\n"
        "    import zlib\n"
        "    from contract import PORT, TRAINING_REQUEST\n"
        "    claim,_=stable_json(D/'CLAIM.json',cap=16384)\n"
        "    approval,actual=stable_json(R/'incoming'/(LABEL+'-authority')/(REQUEST+'.json'),cap=16384)\n"
        "    need(actual['SHA256']==claim['placement_authority_SHA256'] and approval['request_id']==REQUEST,\n"
        "         'R120_COMPUTE_PREFLIGHT_APPROVAL_FILE')\n"
        "    now=time.time()\n"
        "    source=verify('GPU')\n"
        "    target=dict(R=R.as_posix(),PARENT=PARENT,NODE=NODE,PORT=PORT,LABEL=LABEL,\n"
        "                TRAINING_REQUEST=TRAINING_REQUEST,DISPATCH_REQUEST=DISPATCH_REQUEST)\n"
        "    attrs=dict(root=D.as_posix(),manifest=source,target=target,issued=approval['issued_epoch'],\n"
        "               pre_deadline=now+min(seconds,60))\n"
        f"    program=zlib.decompress(b64decode({inline_blob!r},validate=True)).decode()\n"
        "    return [HOST_PYTHON,'-I','-S','-B','-c',program,json.dumps(attrs,sort_keys=True,allow_nan=False)]\n\n\n"
    )
    text=raw.decode()
    text=splice(text,"def execute_step(phase,command,clock,on_handoff=None):",hook+"def execute_step(phase,command,clock,on_handoff=None):")
    text=splice(text,"    argv=step_argv(phase,command,seconds)",
                "    if phase=='pre':\n        command=compute_source_command(clock,seconds)\n    argv=step_argv(phase,command,seconds)")
    return text.encode()


def compact_receiver(raw):
    # Keep the proved receiver shape and controller bytes. Compress only its longer
    # controller member in the wire literal to stay within the old receiver cap.
    source=raw.decode()
    line, = [line for line in source.splitlines() if line.startswith("payload=")]
    payload=ast.literal_eval(line.removeprefix("payload="))
    controller=b64decode(payload["controller.py"],validate=True)
    digest=sha256(controller).hexdigest()
    payload["controller.py"]=b64encode(zlib.compress(controller,9)).decode()
    source=splice(source,line,"payload="+repr(payload))
    source=splice(source,"import os,sys,json,base64,hashlib\n","import os,sys,json,base64,hashlib,zlib\n")
    source=splice(source," p=src/name;assert Path(name).name==name;p.write_bytes(base64.b64decode(value,validate=True))",
         " p=src/name;assert Path(name).name==name;data=base64.b64decode(value,validate=True)\n"
         " if name=='controller.py':\n"
         "  decoder=zlib.decompressobj();data=decoder.decompress(data,65537)\n"
         f"  assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and hashlib.sha256(data).hexdigest()=={digest!r}\n"
         " p.write_bytes(data)")
    out=source.encode()
    compile(out,"<compact exact source receiver>","exec")
    return out


def carrier():
    archived = HERE.parents[2] / "results/bw1000/2026-11-27/training-gemma-r120-successor-original4173-zero-gres-enoent-and-4174-readback.json"
    prior = json.loads(pinned(archived,PREVIOUS_FAILURE))
    require(prior["original_replay_allowed"] is False and prior["actual_gpu_steps"] == prior["actual_optimizer_updates"] == 0
            and prior["request"] == OLD_DISPATCH and prior["scope"] == OLD_SCOPE
            and prior["pre_exact_UTC"][0] == "1103606.4173"
            and prior["later_independent_zero_GRES"][0] == "1103606.4174","ACTUAL_PREVIOUS_VISIBILITY_FAILURE_ARCHIVE")
    original_admission=pinned(PRIOR/"admission.py",ADMISSION_SHA)
    new_admission=moved(original_admission)
    admission=types.ModuleType("_compute_source_ready_admission")
    exec(compile(new_admission,"<rebound exact runtime admission>","exec"),admission.__dict__)
    original=pinned(PRIOR/"prepare.py",CARRIER_SHA)
    # Caller remains the original audited carrier, except the inlined gate source uses the new identity.
    text=splice(original.decode(), '    source=(HERE/"admission.py").read_bytes()',
                '    source=current_admission_source')
    text=splice(text,"    check_receiver(body,result,effective_mf,context)",
                "    check_receiver(body,result,effective_mf,context)\n    body=compact_receiver(body)")
    context=dict(__file__=str(PRIOR/"prepare.py"),__name__="_prior_exact_runtime_carrier")
    exec(compile(text,"<pinned prior R120 carrier>", "exec"),context)
    read_previous=context["stable_members"]
    bootstrap=(HERE/"bootstrap.py").read_bytes()
    compile(bootstrap,"bootstrap.py","exec")

    def stable():
        context["a"]=original_module
        try:
            baseline,schema=read_previous()
        finally:
            context["a"]=admission
        manifest=pinned(PRIOR/"SUCCESSOR_SOURCE_FILES.json",PRIOR_MANIFEST)
        require(context["source_manifest"](baseline,schema)==manifest,"EXACT_PRIOR_SUCCESSOR_CLOSURE")
        blobs={name:moved(raw) for name,raw in baseline.items()}
        blobs["controller.py"]=gated_controller(blobs["controller.py"],bootstrap)
        claim=json.loads(blobs["CLAIM.json"])
        claim.update(source_visibility_prior_evidence_SHA256=PREVIOUS_FAILURE,source_visibility_compute_gate=True,
                     predecessor_replay_allowed=False)
        blobs["CLAIM.json"]=admission.encode(claim)
        require(len(blobs)==32 and {n for n,v in blobs.items() if v!=baseline[n]}==
                {"CLAIM.json","contract.py","controller.py","short_mount.py","verify_original.py"},
                "EXACT_SUCCESSOR_GATE_CHANGED_MEMBERS")
        for name, raw in blobs.items():
            if name.endswith(".py"):
                compile(raw,name,"exec")
        return blobs,schema

    original_module=context["a"]
    context.update(a=admission,HERE=HERE,current_admission_source=new_admission,stable_members=stable,
                   compact_receiver=compact_receiver)
    return context


def main():
    ctx=carrier()
    ctx["main"]()


if __name__=="__main__":
    main()
