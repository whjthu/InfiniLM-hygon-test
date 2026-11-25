"""Package the exact public owner/source closure for the existing private transport."""

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from contract import digest, encode, need, small
from install import contents
from packet_spec import MEMBERS


def build(repo):
    source={name:small(repo/path) for name,path in MEMBERS.items()}
    manifest=encode({"files":{name:{"bytes":len(raw),"SHA256":digest(raw)} for name,raw in source.items()}})
    output=BytesIO()
    with ZipFile(output,"w") as archive:
        for name,data in sorted({**source,"PACKET.json":manifest}.items()):
            info=ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type=ZIP_DEFLATED
            info.create_system=3
            info.external_attr=0o100600<<16
            archive.writestr(info,data)
    raw=output.getvalue()
    sha=digest(manifest)
    contents(raw,sha)
    return raw,sha


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    repo=Path(__file__).resolve().parents[3]
    need(args.out.parent.resolve(strict=True)==repo/"local-only","EXACT_PRIVATE_PACKET_DESTINATION")
    raw,manifest_sha=build(repo)
    with args.out.open("xb") as stream:
        stream.write(raw)
    print(encode(dict(bytes=len(raw),SHA256=digest(raw),manifest_SHA256=manifest_sha,source_files=len(MEMBERS),
                      target_writes=0,GPU=0)).decode().strip())
