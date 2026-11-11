"""Resolve, install into a private prefix, and verify the public Swift package candidate."""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit

root=Path("/replay")
packages=Path("/packages")
started=time.monotonic()
deadline=started+480
manifest=json.loads((root/"PUBLIC_PACKAGES.json").read_bytes())
pinned={row["filename"]:row for row in manifest["packages"]}
for name,row in pinned.items():
    path=packages/name
    assert path.stat().st_size==row["bytes"]
    with path.open("rb") as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
        assert digest.hexdigest()==row["SHA256"]
protected={name:metadata.version(name) for name in ("torch","torchvision")}
constraints=root/"protected-runtime.txt"
with constraints.open("x") as stream:
    stream.write("".join(name+"=="+version+"\n" for name,version in protected.items()))
environment={key:value for key,value in os.environ.items() if not key.startswith("PIP_")}
environment["PIP_CONFIG_FILE"]=os.devnull

def run(label,args,env=environment):
    limit=min(240,max(1,deadline-time.monotonic()-15))
    log=root/(label+".log")
    with log.open("xb") as stream:
        result=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,timeout=limit,env=env)
    raw=log.read_bytes()
    assert len(raw)<1048576
    return {"returncode":result.returncode,"log_SHA256":hashlib.sha256(raw).hexdigest(),
            "errors":[line[-300:] for line in raw.decode(errors="replace").splitlines()
                      if "ERROR:" in line or line.startswith(("ModuleNotFoundError","ImportError","RuntimeError","AttributeError"))][-8:]}

prefix=[sys.executable,"-B","-m","pip","install","--no-index","--find-links",str(packages),
        "--no-input","--disable-pip-version-check","--no-build-isolation"]
result={"protected_distributions":protected,"package_count":len(pinned),"GPU_allocation":0,
        "scope":"Isolated Swift candidate with exact verified public artifacts"}
result["resolution"]=run("offline-plan",[*prefix,"--dry-run","--report",str(root/"pip-plan.json"),
                                        "--constraint",str(constraints),"ms-swift==4.4.2"])
if result["resolution"]["returncode"]==0:
    report=json.loads((root/"pip-plan.json").read_bytes())
    selected=[]
    rows=[]
    for package in report["install"]:
        name=package["metadata"]["name"]
        assert name.lower().replace("_","-") not in ("torch","torchvision")
        uri=urlsplit(package["download_info"]["url"])
        assert uri.scheme=="file"
        path=Path(unquote(uri.path))
        assert path.parent==packages and path.name in pinned
        assert package["download_info"]["archive_info"]["hashes"]["sha256"]==pinned[path.name]["SHA256"]
        selected.append(str(path))
        rows.append({"name":name,"version":package["metadata"]["version"],"filename":path.name,"SHA256":pinned[path.name]["SHA256"]})
    assert selected and not (root/"site").exists()
    result["selected_packages"]=rows
    result["install"]=run("offline-install",[*prefix,"--no-deps","--target",str(root/"site"),*selected])
    if result["install"]["returncode"]==0:
        verify_env={**environment,"PYTHONPATH":str(root/"site")+os.pathsep+environment.get("PYTHONPATH","")}
        result["import_check"]=run("offline-import-check",[sys.executable,"-B",str(root/"inventory_runtime.py")],verify_env)
        path=root/"inventory.json"
        if path.is_file():
            raw=path.read_bytes()
            result["inventory_SHA256"]=hashlib.sha256(raw).hexdigest()
            result["inventory"]=json.loads(raw)
result["base_runtime_versions_unchanged"]=all(metadata.version(name)==version for name,version in protected.items())
assert result["base_runtime_versions_unchanged"]
result["qualified_SFT_import"]=bool(result.get("inventory",{}).get("swift_SFT_import_available")) and not result.get("inventory",{}).get("missing_or_incompatible",[True])
result["wall_seconds"]=time.monotonic()-started
with (root/"summary.json").open("x") as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result))
