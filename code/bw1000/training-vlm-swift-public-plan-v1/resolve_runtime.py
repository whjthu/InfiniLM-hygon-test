"""Resolve an isolated Swift environment against the actual installed DTK packages."""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path("/replay")
started=time.monotonic()
constraints=root/"protected-runtime.txt"
protected={name:metadata.version(name) for name in ("torch","torchvision")}
with constraints.open("x",encoding="utf-8") as stream:
    stream.write("".join(name+"=="+version+"\n" for name,version in protected.items()))
ca=next((p for p in ("/etc/pki/tls/certs/ca-bundle.crt","/etc/ssl/certs/ca-certificates.crt")
         if Path(p).is_file()),None)
assert ca
environment={key:value for key,value in os.environ.items() if not key.startswith("PIP_")}
environment["PIP_CONFIG_FILE"]=os.devnull
args=[sys.executable,"-B","-m","pip","install","--dry-run","--report",str(root/"pip-plan.json"),
      "--no-input","--disable-pip-version-check","--index-url","https://pypi.org/simple",
      "--cert",ca,"--timeout","12","--retries","1","--constraint",str(constraints),"ms-swift==4.4.2"]
with (root/"pip-plan.log").open("xb") as log:
    outcome=subprocess.run(args,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=250)
raw=(root/"pip-plan.log").read_bytes()
assert len(raw)<1048576
result={"returncode":outcome.returncode,"protected_distributions":protected,"base_torch_unchanged":
        all(metadata.version(name)==version for name,version in protected.items()),
        "dry_run":True,"base_environment_installation_performed":False,"wall_seconds":time.monotonic()-started,
        "log_SHA256":hashlib.sha256(raw).hexdigest(),
        "errors":[line[-300:] for line in raw.decode(errors="replace").splitlines()
                  if "ERROR:" in line or "Could not fetch URL" in line][-8:]}
report_path=root/"pip-plan.json"
if report_path.is_file():
    data=report_path.read_bytes()
    report=json.loads(data)
    assert report["environment"]["python_version"]=="3.10"
    packages=[]
    from urllib.parse import urlsplit
    for row in report["install"]:
        name,version=row["metadata"]["name"],row["metadata"]["version"]
        url=row["download_info"]["url"]
        assert urlsplit(url).hostname in ("pypi.org","files.pythonhosted.org")
        digest=row["download_info"]["archive_info"]["hashes"]["sha256"]
        assert len(digest)==64
        assert name.lower().replace("_","-") not in ("torch","torchvision")
        packages.append({"name":name,"version":version,"url":url,"SHA256":digest,
                         "requires_dist":row["metadata"].get("requires_dist",[]),"requested":row["requested"]})
    result.update(report_SHA256=hashlib.sha256(data).hexdigest(),packages=packages,
                  resolution_environment=report["environment"])
with (root/"summary.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result))
