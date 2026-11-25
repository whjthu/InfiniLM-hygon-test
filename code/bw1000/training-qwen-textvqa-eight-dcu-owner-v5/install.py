"""Atomic source-only staging for a previously verified packet on the shared HOST."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import sys
from zipfile import ZipFile

from contract import CLOSED_REQUESTS, PREFIX, REQUEST, RESULTS, SHA64, digest, encode, need, save, small
from packet_spec import MEMBERS


def contents(raw, manifest_sha):
    with ZipFile(BytesIO(raw)) as packet:
        names=packet.namelist()
        need(len(names)==len(MEMBERS)+1 and set(names)=={"PACKET.json",*MEMBERS},"EXACT_SOURCE_PACKET_FILE_SET")
        out={}
        for info in packet.infolist():
            name=info.filename
            need(not info.is_dir() and info.orig_filename==name and not info.flag_bits&1
                 and stat.S_IFMT(info.external_attr>>16) in {0,stat.S_IFREG}
                 and 0<info.file_size<=262144,"EXACT_SOURCE_PACKET_FILE_EXTENT")
            with packet.open(info) as stream:
                out[name]=stream.read(info.file_size+1)
            need(len(out[name])==info.file_size,"EXACT_SOURCE_PACKET_MEMBER_BYTES")
    manifest=out["PACKET.json"]
    need(digest(manifest)==manifest_sha,"EXACT_SOURCE_PACKET_MANIFEST")
    pins=json.loads(manifest)["files"]
    need(set(pins)==set(MEMBERS),"EXACT_SOURCE_PACKET_MANIFEST_MEMBERSHIP")
    for name,pin in pins.items():
        value=out[name]
        need(pin==dict(bytes=len(value),SHA256=digest(value)),"EXACT_SOURCE_PACKET_STAGED_SHA")
        if name.endswith(".py"):
            compile(value,name,"exec")
    return out


def install(results,request,raw,source_sha):
    need(REQUEST.fullmatch(request) and request not in CLOSED_REQUESTS
         and SHA64.fullmatch(source_sha) and 0<len(raw)<=524288,
         "EXACT_SOURCE_INSTALL_REQUEST")
    files=contents(raw,source_sha)
    run=results/(PREFIX+request)
    run.mkdir(mode=0o700)
    tentative=run/"source-installing"
    tentative.mkdir(mode=0o700)
    for name,value in files.items():
        path=tentative/name
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        need(small(path)==value,"EXACT_INSTALLED_SOURCE_READBACK")
    staged=run/"src"
    tentative.rename(staged)
    for name,value in files.items():
        need(small(staged/name)==value,"EXACT_ATOMIC_SOURCE_READBACK")
    report={"request":request,"source_manifest_SHA256":source_sha,"source_files":len(files)-1,
            "state":"EXACT_SOURCE_ONLY_PACKET_INSTALLED","GPU":0,"scheduler_operations":0}
    save(run/"install.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",required=True)
    parser.add_argument("--packet",type=Path,required=True)
    parser.add_argument("--packet-sha",required=True)
    parser.add_argument("--source-sha",required=True)
    args=parser.parse_args()
    need(sys.platform=="linux" and SHA64.fullmatch(args.packet_sha),"ACTUAL_ORIGINAL_SHARED_HOST_ONLY")
    os.umask(0o077)
    print(encode(install(RESULTS,args.request,small(args.packet,args.packet_sha),args.source_sha)).decode().strip())
