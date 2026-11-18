"""Only new installer metadata validation; no mkdir/exec/Slurm/weight or old fixtures."""
import ast
import base64
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

D=Path(__file__).resolve().parent


def main():
    module=ast.parse((D/'install_and_run.py').read_bytes())
    funcs=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('need','validate_payload')]
    scope=dict(base64=base64,hashlib=hashlib,json=json,math=math,PurePosixPath=PurePosixPath,re=re,time=time,
               ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results'),LABEL=D.name)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'actual-installer-metadata','exec'),scope)
    manifest=json.loads((D/'SOURCE_FILES.json').read_bytes())
    raw={name:(D/name).read_bytes() for name in manifest['files']}
    # A bounded in-memory adoption fixture changes only copied claim bytes and its MF pin.
    claim=json.loads(raw['CLAIM.json']);claim.update(root_GO=True,CPU_GO=True)
    raw['CLAIM.json']=(json.dumps(claim,indent=2)+'\n').encode()
    def package(values):
        mf=dict(files={n:dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest()) for n,b in values.items()})
        body=(json.dumps(mf,indent=2)+'\n').encode();members=dict(values,**{'SOURCE_FILES.json':body})
        return dict(epoch=time.time()-1,request_id='a'*32,publication='b'*40,
            source_manifest_SHA256=hashlib.sha256(body).hexdigest(),
            package={n:base64.b64encode(b).decode() for n,b in members.items()})
    payload=package(raw);assert len(scope['validate_payload'](payload))==len(raw)+1
    rejected=[]
    variants=[]
    v=copy.deepcopy(payload);v['epoch']=time.time()-901;variants.append(('expired-original-epoch',v))
    v=copy.deepcopy(payload);v['package']['../outside.py']=base64.b64encode(b'x').decode();variants.append(('path-traversal',v))
    v=copy.deepcopy(payload);v['package']['worker.py']=base64.b64encode(b'corrupt').decode();variants.append(('changed-source-body',v))
    bad=copy.deepcopy(raw);c=json.loads(bad['CLAIM.json']);c['GPU_GO']=True;bad['CLAIM.json']=json.dumps(c).encode();variants.append(('CPU-installer-cannot-enable-GPU',package(bad)))
    for name,value in variants:
        try:scope['validate_payload'](value)
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    print(json.dumps(dict(passed=True,valid_metadata_only=1,rejected=rejected,actual_install_mkdir_exec=0,
                          Slurm_GPU_calls=0,weight_reads=0)))


if __name__=='__main__':main()
