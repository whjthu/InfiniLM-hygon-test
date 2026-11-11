"""Independently rehash the original MiniCPM CPU template evidence on A100-27."""
import hashlib,json,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parent
config=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config")
code=r"""
import datetime,hashlib,json
from pathlib import Path
root=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-original-processor-v1")
raw=(root/"receipt.json").read_bytes();receipt=json.loads(raw)
assert receipt["claim_commit"]=="1368fe6a3ffb48227a77b23a38d2dbc69d1fa957"
assert receipt["state"]=="CPU_ORIGINAL_PROCESSOR_COMPLETE" and receipt["rows"]==64
assert receipt["model_loaded"] is False and receipt["device_initialized"] is False
ident=(root/"encoded-identities.json").read_bytes()
assert len(ident)==receipt["encoded_identities"]["bytes"] and hashlib.sha256(ident).hexdigest()==receipt["encoded_identities"]["SHA256"]
rows=json.loads(ident)
assert [x["row_ordinal"] for x in rows]==list(range(1,65))
lengths=[];labeled=[];tensor_count=0
def tensors(value):
    if isinstance(value,dict):
        if value.get("kind")=="tensor":return 1
        return sum(tensors(x) for x in value.values())
    if isinstance(value,list):return sum(tensors(x) for x in value)
    return 0
for row in rows:
    encoded=row["encoded"]
    assert len(encoded["input_ids"])==len(encoded["labels"])<=4096
    lengths.append(len(encoded["input_ids"]));labeled.append(sum(x!=-100 for x in encoded["labels"]))
    tensor_count+=tensors(encoded)
assert min(labeled)>0
payload=Path(receipt["payload"]["path"])
assert payload==root/"encoded-original.pt" and payload.stat().st_size==receipt["payload"]["bytes"]
h=hashlib.sha256()
with payload.open("rb") as f:
    for data in iter(lambda:f.read(1024*1024),b""):h.update(data)
assert h.hexdigest()==receipt["payload"]["SHA256"]
print(json.dumps({"observed_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"receipt_SHA256":hashlib.sha256(raw).hexdigest(),
    "receipt":receipt,"rows":64,"encoded_tensor_fields":tensor_count,"input_length_min":min(lengths),"input_length_max":max(lengths),
    "input_tokens_total":sum(lengths),"labeled_tokens_total":sum(labeled),"all_rows_have_supervised_tokens":True,"payload_independently_rehashed":True}))
"""
r=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(config),"a100-27","python3"],input=code.encode(),capture_output=True,timeout=60)
with (BASE/"independent-reader.stderr").open("xb") as f:f.write(r.stderr)
assert r.returncode==0
v=json.loads(r.stdout)
with (BASE/"independent-processor-original.json").open("x") as f:json.dump(v,f,indent=2)
print(json.dumps(v))
