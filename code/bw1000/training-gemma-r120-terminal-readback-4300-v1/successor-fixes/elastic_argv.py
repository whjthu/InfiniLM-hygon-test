"""Narrowly insert a private log root in a verified native torchrun argv."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import os
import runpy
import stat
import sys


SWIFT_CLI_SHA="d84a44f9911c01c9af83db19baf2068ad39a8ec926f6e2c58f6f69779bfdad68"
SWIFT_ENTRY_SHA="8f72975bdfe86477b95da8d51ed34f755aff9012f05f33072ae128bc6c6829cb"
SFT_PATH="/swift-runtime/site/swift/cli/sft.py"
SFT_SHA="97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0"
DISPATCH_WITNESS="3ce5eddfc2eac1a08a58bec5ea4b718809090a5c11f89ef429e072b1f1d1190c"


def need(ok,code):
    if not ok:
        raise ValueError(code)


def file_pin(path, size, digest):
    with Path(path).open("rb") as stream:
        raw=stream.read(size+1)
    need(len(raw)==size and sha256(raw).hexdigest()==digest,"EXACT_ELASTIC_PRODUCER_SOURCE")
    return raw


def log_root(run, short):
    base=Path(run)
    private=base/"native-torchelastic-logs"
    need(base.is_absolute() and base.resolve(strict=True)==base and not base.is_relative_to(Path(short))
         and private.parent==base and not private.is_relative_to(Path(short)), "PRIVATE_LOG_ROOT_OUTSIDE_SHORT_IPC")
    m=base.lstat()
    need(stat.S_ISDIR(m.st_mode) and m.st_uid==os.getuid() and stat.S_IMODE(m.st_mode)==0o700
         and not os.path.lexists(private),"NEW_OWNER_ONLY_ELASTIC_LOG_ROOT")
    private.mkdir(mode=0o700)
    return private


def pin_dispatch(body, proof):
    need(sha256(proof).hexdigest()==DISPATCH_WITNESS,"EXACT_ORIGINAL_SWIFT_DISPATCHER_WITNESS")
    evidence=json.loads(proof)
    need(evidence["source_SHA256"]==SWIFT_CLI_SHA and len(body)==3264
         and sha256(body).hexdigest()==SWIFT_CLI_SHA,"EXACT_ORIGINAL_SWIFT_DISPATCHER_BYTES")
    tree=ast.parse(body)
    wanted={"get_torchrun_args","cli_main"}
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted}
    need(set(functions)==wanted and all(ast.dump(n,include_attributes=False)==
         ast.dump(ast.parse(evidence["members"][name]).body[0],include_attributes=False)
         for name,n in functions.items()),"EXACT_PROVEN_SWIFT_TORCHRUN_ASSEMBLY")


class Dispatch:
    def __init__(self,fn,private,request,port):
        self.fn,self.namespace=fn,fn.__globals__
        self.private,self.request,self.port=private,request,port
        self.getter=self.namespace["get_torchrun_args"]
        self.subprocess=self.namespace["subprocess"]
        self.prefix=None
        self.sent=False

    def arguments(self):
        need(self.prefix is None,"ONLY_ONE_NATIVE_TORCHRUN_GETTER")
        result=self.getter()
        expected=[]
        for key in ("NPROC_PER_NODE","MASTER_PORT","NNODES","NODE_RANK","MASTER_ADDR"):
            if key in os.environ:
                expected.extend(("--"+key.lower(),os.environ[key]))
        need(result==expected and os.environ.get("NPROC_PER_NODE")=="8"
             and os.environ.get("MASTER_PORT")==str(self.port)
             and os.environ.get("MASTER_ADDR")=="127.0.0.1",
             "EXACT_PRODUCTION_TORCHRUN_ARGS")
        self.prefix=[*result,"--log-dir",str(self.private)]
        return self.prefix[:]

    def run(self,argv):
        need(not self.sent and self.prefix is not None and type(argv) is list
             and argv==[sys.executable,"-m","torch.distributed.run",*self.prefix,SFT_PATH,*sys.argv[2:]]
             and argv.count("--log-dir")==1,"EXACT_ONE_ASSEMBLED_NATIVE_TORCHRUN")
        pin=sha256(json.dumps(argv,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
        receipt=dict(request=self.request,argv_SHA256=pin,log_root=self.private.name,
                     exact_swift_dispatcher_SHA256=SWIFT_CLI_SHA,log_dir_is_sole_change=True,rank_count=8)
        with (self.private.parent/"native-torchelastic-intent.json").open("x",encoding="utf8") as stream:
            json.dump(receipt,stream,sort_keys=True,allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.sent=True
        return self.subprocess.run(argv)

    def invoke(self):
        self.namespace["get_torchrun_args"]=self.arguments
        self.namespace["subprocess"]=self
        try:
            result=self.fn()
            need(self.sent,"MISSING_NATIVE_TORCHRUN_INVOCATION")
            return result
        finally:
            self.namespace["get_torchrun_args"]=self.getter
            self.namespace["subprocess"]=self.subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original",type=Path)
    parser.add_argument("swift",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    from contract import ASSET, D, PORT, REQUEST, ROOT, verify
    from short_mount import SHORT
    verify("GPU")
    original=ASSET/"framework/vlm/runtime/swift_entry.py"
    need(args.original==original and args.swift[:1]==["sft"],"EXACT_ORIGINAL_SWIFT_SFT_ENTRY")
    raw=original.read_bytes()
    need(0<len(raw)<=16384 and sha256(raw).hexdigest()==SWIFT_ENTRY_SHA,"EXACT_CANONICAL_SWIFT_ENTRY_BYTES")
    preload=runpy.run_path(str(original),run_name="_frozen_swift_entry")
    fn=preload["resolve_cli"]()
    body=file_pin(Path(fn.__globals__["__file__"]),3264,SWIFT_CLI_SHA)
    pin_dispatch(body,(D/"SWIFT_DISPATCH_WITNESS.json").read_bytes())
    file_pin(SFT_PATH,583,SFT_SHA)
    private=log_root(ROOT,SHORT)
    sys.argv=[str(original),*args.swift]
    return Dispatch(fn,private,REQUEST,PORT).invoke()


if __name__=="__main__":
    raise SystemExit(main())
