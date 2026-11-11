"""Read exact original Hygon CPU outcome and independently verify its evidence."""
import hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
plan=json.loads((BASE/"PROCESSOR_INPUTS.json").read_bytes())
prior=json.loads((BASE/"PRIOR_INPUTS.json").read_bytes())
ns={"__file__":str(Path(__file__).resolve())}
prelude=(BASE.parent/"i03-deep-c4-history-staging-v1/read_source_once.py").read_text(encoding="utf8")
exec(compile(prelude.split("REMOTE=",1)[0],"existing_cluster_ssh","exec"),ns)
REMOTE=r"""
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-hygon-processor-r2")
work=root/"cpu"
def read(p):
    raw=p.read_bytes()
    assert len(raw)<1500000
    return {"SHA256":hashlib.sha256(raw).hexdigest(),"value":json.loads(raw)}
outcome=read(work/"outcome.json");value=outcome["value"]
assert value["claim_commit"]=="901c1a3740d3888883a7a17694108c33e18a8274"
accounting=subprocess.check_output(["sacct","-n","-P","-j","1087656","--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw"],text=True,timeout=12)
rows=[x.split("|") for x in accounting.splitlines() if "|training-minicpm-hygon-processor-r2|" in x]
assert len(rows)==1 and rows==value["accounting"]
assert rows[0][2] in ("COMPLETED","FAILED","CANCELLED","OUT_OF_MEMORY","TIMEOUT","NODE_FAIL")
assert value["deadline"]-value["epoch"]==600
verified=[]
for item in PLAN["files"]:
    p=root/item["name"];assert p.is_file() and not p.is_symlink()
    raw=p.read_bytes()
    assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["SHA256"]
    verified.append(item["name"])
prior_verified=0
for item in PRIOR["files"]:
    p=root.parent/"training-minicpm-hygon-processor-v1"/item["name"]
    raw=p.read_bytes()
    assert len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["SHA256"]
    prior_verified+=1
log=(work/"CPU.log").read_bytes()
assert hashlib.sha256(log).hexdigest()==value["log_SHA256"]
result={"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"accounting":rows,
        "original_outcome":outcome,"input_files_rehashed":len(verified),"prior_input_files_rehashed":prior_verified,"CPU_epoch_seconds":600,
        "GPU_execution":False,"first_update_numerics_qualified":False}
for name in ("overlay.json","encoded/failure.json","encoded/receipt.json","comparison.json"):
    p=work/name
    if p.is_file():
        record=read(p);assert record==value[name]
        result[name]=record
if (work/"encoded/receipt.json").is_file():
    receipt=result["encoded/receipt.json"]["value"]
    assert receipt["model_loaded"] is False and receipt["device_initialized"] is False
    assert receipt["rows"]==64
    payload=work/"encoded/encoded-original.pt"
    assert payload.stat().st_size==receipt["payload"]["bytes"]
    h=hashlib.sha256()
    with payload.open("rb") as f:
        for raw in iter(lambda:f.read(1024*1024),b""):h.update(raw)
    assert h.hexdigest()==receipt["payload"]["SHA256"]
    actual=(work/"encoded/encoded-identities.json").read_bytes()
    expected=(root.parent/"training-minicpm-hygon-processor-v1/expected-identities.json").read_bytes()
    assert hashlib.sha256(actual).hexdigest()==receipt["encoded_identities"]["SHA256"]
    result.update(payload_independently_rehashed=True,encoded_identity_documents_exact=actual==expected)
print(json.dumps(result))
"""
r=subprocess.run(ns["ssh"],input=("PLAN="+repr(plan)+"\nPRIOR="+repr(prior)+"\n"+REMOTE).encode(),capture_output=True,timeout=120)
with (BASE/"independent-reader.stderr").open("xb") as f:f.write(r.stderr)
assert r.returncode==0,f"reader rc={r.returncode}"
value=json.loads(r.stdout)
with (BASE/"independent-original-CPU.json").open("x") as f:json.dump(value,f,indent=2)
keys=("observed_UTC","accounting","input_files_rehashed","prior_input_files_rehashed","CPU_epoch_seconds","payload_independently_rehashed","encoded_identity_documents_exact","encoded/failure.json","comparison.json")
print(json.dumps({k:value[k] for k in keys if k in value}))
