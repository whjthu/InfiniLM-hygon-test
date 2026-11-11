"""Compare all actual native CPU encoded fields against the original A100 capture."""
import hashlib,json
from pathlib import Path
ROOT=Path("/replay")
expected_raw=Path("/processor-reference/expected-identities.json").read_bytes()
assert hashlib.sha256(expected_raw).hexdigest()=="dd2b6af2c93d88281fa7b4cda2d6c4cc6ed2bec8780f8c32ca73790a43763235"
actual_raw=(ROOT/"encoded/encoded-identities.json").read_bytes()
original=json.loads(expected_raw);actual=json.loads(actual_raw)
assert len(original)==len(actual)==64
differences=[];tensor_count=0
def walk(a,b,path):
    global tensor_count
    if type(a)!=type(b):
        differences.append({"path":path,"reason":"type"});return
    if isinstance(a,dict):
        if a.get("kind")=="tensor":tensor_count+=1
        if a.keys()!=b.keys():
            differences.append({"path":path,"reason":"keys"});return
        for k in a:walk(a[k],b[k],path+"/"+str(k))
    elif isinstance(a,list):
        if len(a)!=len(b):
            differences.append({"path":path,"reason":"length"});return
        for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+"/"+str(i))
    elif a!=b:differences.append({"path":path,"reason":"value"})
for i,(a,b) in enumerate(zip(original,actual)):
    assert a["row_ordinal"]==b["row_ordinal"]==i+1
    walk(a,b,str(i+1))
receipt=json.loads((ROOT/"encoded/receipt.json").read_bytes())
assert receipt["model_loaded"] is False and receipt["device_initialized"] is False
summary={"state":"CPU_ENCODED_FIELDS_EXACT" if not differences else "CPU_ENCODED_FIELDS_DIFFER",
    "rows":64,"expected_tensor_fields":tensor_count,"different_fields":len(differences),
    "difference_paths":differences[:64],"all_fields_exact":not differences,
    "original_identities_SHA256":hashlib.sha256(expected_raw).hexdigest(),
    "actual_identities_SHA256":hashlib.sha256(actual_raw).hexdigest(),
    "capture_receipt":receipt,"first_update_numerics_qualified":False,"GPU_execution":False}
with (ROOT/"comparison.json").open("x") as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
