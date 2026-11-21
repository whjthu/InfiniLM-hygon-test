"""Acquire one bounded, source-pinned VGGT training pair from public Freiburg3 data."""
from bisect import bisect_left
from collections import Counter
import ctypes
from decimal import Decimal
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import tarfile
import urllib.request
import zlib


SCENE = "nostructure_texture_far"
SEQUENCE = "freiburg3_" + SCENE
R = Path(__file__).resolve().parent
ARCHIVE_ROOT = "rgbd_dataset_" + SEQUENCE
TUM = "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/" + ARCHIVE_ROOT
REV = "f21e73c5bf0d91fc66f3c5759c9d438e52ab24a6"
MREL = "data/freiburg3/fr3_" + SCENE + ".fo.mcap"
HF = "https://huggingface.co/datasets/Voxel51/TUM-RGBD/"
MCAP = HF + "resolve/" + REV + "/" + MREL
LIMIT = 8 * 1024 * 1024
B = 10**9
REPORT = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(name, payload):
    path = R / name
    path.parent.mkdir(exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def request(url, cap, byte_range=None):
    headers = {"Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = byte_range
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=22) as response:
        if byte_range:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match:
                raise ValueError("Range was not served without downloading object")
            first,last,total = map(int,match.groups())
            if (first, last) == (0,total-1) or last-first+1>cap:
                raise ValueError("Range outside small-sample cap")
            if byte_range != f"bytes={first}-{last}" and byte_range != f"bytes=-{last-first+1}":
                raise ValueError("Unexpected returned range")
            cap = last-first+1
        elif response.status != 200:
            raise ValueError("Unexpected metadata status")
        raw=response.read(cap+1)
        if not raw or len(raw)>cap or (byte_range and len(raw)!=cap):
            raise ValueError("Invalid bounded response length")
    receipt={"url":url,"bytes":len(raw),"SHA256":sha(raw)}
    if byte_range:
        receipt.update(start=first,end=last,total=total)
    REPORT.setdefault("retrievals",[]).append(receipt)
    return raw, receipt


def ns(value):
    converted=Decimal(value)*B
    if converted!=int(converted):
        raise ValueError("Invalid timestamp precision")
    return int(converted)


def path(stamp, kind):
    if stamp%1000:
        raise ValueError("Frame timestamp differs from TUM microsecond PNG convention")
    return f"{kind}/{stamp//B}.{stamp%B//1000:06d}.png"


def record(buf,pos):
    op,size=struct.unpack_from("<BQ",buf,pos)
    end=pos+9+size
    if end>len(buf):
        raise ValueError("Invalid MCAP record extent")
    return op,buf[pos+9:end],end


def string(buf,pos):
    size,=struct.unpack_from("<I",buf,pos)
    if size+pos+4>len(buf):
        raise ValueError("Invalid MCAP string")
    return buf[pos+4:pos+4+size].decode(),pos+4+size


def summary():
    pointer,_=request(HF+"raw/"+REV+"/"+MREL,1024)
    expected=re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\r?\noid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)\r?\n?",pointer)
    if not expected:
        raise ValueError("Pinned mirror LFS pointer malformed")
    REPORT["mirror_published_object_SHA256"]=expected[1].decode()
    total=int(expected[2])
    REPORT["mirror_published_object_bytes"]=total
    tail,_=request(MCAP+"?download=true",37,"bytes=-37")
    if tail[-8:]!=b"\x89MCAP0\r\n" or struct.unpack_from("<BQ",tail)!=(2,20):
        raise ValueError("Invalid MCAP Footer")
    begin,offset,crc=struct.unpack_from("<QQI",tail,9)
    if not begin<offset<total-37 or total-37-begin>128*1024:
        raise ValueError("Summary outside cap")
    raw,head=request(MCAP+"?download=true",128*1024,f"bytes={begin}-{total-38}")
    if head["total"]!=total or not crc or zlib.crc32(raw+tail[:25])!=crc:
        raise ValueError("Invalid MCAP Summary CRC")
    chunks=[]
    ids={}
    schema={}
    pos=0
    while pos<offset-begin:
        op,p,pos=record(raw,pos)
        if op==3:
            sid,=struct.unpack_from("<H",p)
            name,_=string(p,2)
            schema[sid]=name
        elif op==4:
            cid,sid=struct.unpack_from("<HH",p)
            name,_=string(p,4)
            ids[name]=(cid,sid)
        elif op==8:
            first,last,base,length,maplen=struct.unpack_from("<QQQQI",p)
            indexes=dict(struct.iter_unpack("<HQ",p[36:36+maplen]))
            indexlen,=struct.unpack_from("<Q",p,36+maplen)
            codec,end=string(p,44+maplen)
            _,unpacked=struct.unpack_from("<QQ",p,end)
            ceiling=base+length+indexlen
            if not all(base+length<=ix<ceiling<=begin for ix in indexes.values()):
                raise ValueError("Message Index outside expected region")
            chunks.append((first,last,base,length,indexes,codec,unpacked,ceiling))
    cid,sid=ids["/depth"]
    if schema[sid]!="foxglove.CompressedImage":
        raise ValueError("Unexpected depth schema")
    return cid,chunks


def depth_index(cid,chunks,rgb):
    selected={}
    for first,last,base,length,indexes,codec,unpacked,ceiling in chunks:
        if cid not in indexes or not any(first-55000000<=t<=last+55000000 for t in rgb):
            continue
        start=indexes[cid]
        stop=min((o for o in indexes.values() if o>start),default=ceiling)
        data,_=request(MCAP+"?download=true",20000,f"bytes={start}-{stop-1}")
        op,p,end=record(data,0)
        channel,size=struct.unpack_from("<HI",p)
        if op!=7 or channel!=cid or size!=len(p)-6 or end!=len(data):
            raise ValueError("Invalid depth Message Index")
        for t,o in struct.iter_unpack("<QQ",p[6:]):
            if t in selected or not first<=t<=last or not o<unpacked:
                raise ValueError("Duplicate or invalid depth entry")
            selected[t]=(base,length,codec,unpacked,o)
    return selected


def nearest(times,t):
    pos=bisect_left(times,t)
    if not 0<pos<len(times):
        return None
    found=min(times[pos-1:pos+1],key=lambda x:(abs(x-t),x))
    return found if abs(found-t)<=20000000 else None


def choose(images,poses,depth):
    times=sorted(images)
    depth_times=sorted(depth)
    pose_times=sorted(poses)
    for i,first in enumerate(times):
        options=sorted((x for x in times[i+1:] if 100000000<=x-first<=500000000),
                       key=lambda x:(abs(x-first-250000000),x))
        for second in options:
            d1,d2=(nearest(depth_times,t) for t in (first,second))
            p1,p2=(nearest(pose_times,t) for t in (first,second))
            if None not in (d1,d2,p1,p2) and d1!=d2 and math.dist(poses[p1],poses[p2])>=0.005:
                return (first,second),(d1,d2),(p1,p2)
    raise ValueError("No fixed first eligible pair among original RGB prefix members")


def protobuf(data):
    def varint(pos):
        acc=0
        for shift in range(0,70,7):
            value=data[pos]
            pos+=1
            acc|=(value&127)<<shift
            if value<128:
                return acc,pos
        raise ValueError("Oversize protobuf varint")
    fields={}
    pos=0
    while pos<len(data):
        key,pos=varint(pos)
        number,wire=key>>3,key&7
        if not number or number in fields:
            raise ValueError("Invalid protobuf field")
        if wire==0:
            value,pos=varint(pos)
        elif wire==2:
            size,pos=varint(pos)
            value=data[pos:pos+size]
            pos+=size
            if pos>len(data):
                raise ValueError("Truncated protobuf field")
        else:
            raise ValueError("Invalid protobuf wire type")
        fields[number]=value
    return fields


def depth_png(timestamp,cid,pin,cache):
    base,length,declared,unpacked,offset=pin
    if base not in cache:
        raw,_=request(MCAP+"?download=true",3*1024*1024,f"bytes={base}-{base+length-1}")
        op,p,end=record(raw,0)
        first,last,size,crc=struct.unpack_from("<QQQI",p)
        codec,pos=string(p,28)
        compressed,=struct.unpack_from("<Q",p,pos)
        encoded=p[pos+8:]
        if (op!=6 or end!=len(raw) or size!=unpacked or not size<=4*1024*1024
                or codec!=declared or compressed!=len(encoded)):
            raise ValueError("Invalid depth Chunk")
        if codec=="zstd":
            lib=ctypes.CDLL(r"C:\Program Files\Git\mingw64\bin\libzstd.dll")
            lib.ZSTD_decompress.argtypes=(ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t)
            lib.ZSTD_decompress.restype=ctypes.c_size_t
            dst=ctypes.create_string_buffer(size)
            if lib.ZSTD_decompress(dst,size,encoded,len(encoded))!=size:
                raise ValueError("Unexpected Zstd decode")
            decoded=dst.raw
        elif codec=="":
            decoded=encoded
        else:
            raise ValueError("Unexpected depth compression")
        if not crc or crc!=zlib.crc32(decoded):
            raise ValueError("Unexpected depth Chunk CRC")
        cache[base]=decoded
    op,p,_=record(cache[base],offset)
    channel,seq,log,publish=struct.unpack_from("<HIQQ",p)
    fields=protobuf(p[22:])
    clock=protobuf(fields[1])
    if op!=5 or channel!=cid or log!=publish or log!=timestamp or clock[1]*B+clock.get(2,0)!=timestamp:
        raise ValueError("Depth timestamps disagree")
    raw=fields[2]
    if (fields[3]!=b"png" or raw[:8]!=b"\x89PNG\r\n\x1a\n"
            or struct.unpack_from(">IIBB",raw,16)!=(640,480,16,0)):
        raise ValueError("Invalid registered 16-bit depth frame")
    return raw


def acquire():
    prefix,head=request(TUM+".tgz",LIMIT,f"bytes=0-{LIMIT-1}")
    poses_raw,_=request(TUM+"-groundtruth.txt",1024*1024)
    rgb,official_depth={},{}
    names=Counter()
    try:
        with tarfile.open(fileobj=io.BytesIO(prefix),mode="r|gz") as archive:
            for member in archive:
                match=re.fullmatch(ARCHIVE_ROOT+r"/(rgb|depth)/(\d+\.\d+)\.png",member.name)
                if match and member.isfile():
                    if not 0<member.size<2*1024*1024:
                        raise ValueError("Unexpected TUM frame size")
                    raw=archive.extractfile(member).read()
                    if len(raw)!=member.size:
                        raise EOFError("Incomplete boundary member")
                    kind,stamp=match.groups()
                    names[kind]+=1
                    (rgb if kind=="rgb" else official_depth)[ns(stamp)]=raw
    except (EOFError,tarfile.ReadError):
        pass
    REPORT["bounded_tgz_complete_members"]=dict(names)
    if len(rgb)<2:
        raise ValueError("Official 8 MiB prefix did not deliver two complete RGB PNG")
    poses={}
    for row in poses_raw.decode().splitlines():
        cells=row.partition("#")[0].split()
        if cells:
            poses[ns(cells[0])]=tuple(map(float,cells[1:4]))
    REPORT["official_pose_rows"]=len(poses)
    cid,chunks=summary()
    depths=depth_index(cid,chunks,rgb)
    REPORT["depth_index_rows"]=len(depths)
    chosen,d_stamps,p_stamps=choose(rgb,poses,depths)
    REPORT["frozen_pair_timestamps_ns"]={"rgb":chosen,"depth":d_stamps,"pose":p_stamps}
    selected={"groundtruth.txt":poses_raw}
    report_depth=[]
    cache={}
    for stamp in d_stamps:
        raw=depth_png(stamp,cid,depths[stamp],cache)
        original=official_depth.get(stamp)
        if original is not None and original!=raw:
            raise ValueError("Depth conversion differs from original PNG")
        selected[path(stamp,"depth")]=raw
        report_depth.append({"timestamp_ns":stamp,"matched_official_png":original is not None})
    REPORT["depth_direct_official_comparison"]=report_depth
    for stamp in chosen:
        selected[path(stamp,"rgb")]=rgb[stamp]
    def render(times,kind):
        return ("# Timed frame identity from source members, see SOURCE.json\n" +
                "".join(f"{path(t,kind).split('/')[1][:-4]} {path(t,kind)}\n" for t in times)).encode()
    selected["rgb.txt"]=render(chosen,"rgb")
    selected["depth.txt"]=render(sorted(depths),"depth")
    for name,raw in selected.items():
        write(name,raw)
    receipt={"sequence":SEQUENCE,"rgb_indices":[0,1],"files_SHA256":{name:sha(raw) for name,raw in selected.items()}}
    write("pair-manifest.json",(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode())
    REPORT["files_bytes"]={name:len(raw) for name,raw in selected.items()}
    REPORT["status"]="SOURCE_PAIR_PINNED"


if __name__=="__main__":
    try:
        acquire()
    except Exception as error:
        REPORT["status"]="UNAVAILABLE"
        REPORT["failure"]=f"{type(error).__name__}: {str(error)[:180]}"
    write("SOURCE.json",(json.dumps(REPORT,sort_keys=True,indent=2)+"\n").encode())
    print(json.dumps(REPORT,sort_keys=True))
    raise SystemExit(REPORT["status"]!="SOURCE_PAIR_PINNED")
