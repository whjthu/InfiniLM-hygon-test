"""Observe an already-authorized Pack16 compute step before the HIP gate."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import re
import subprocess


FILENAME="START_STEP_QUERY_EVIDENCE.json"
FIELDS=("StepId","Name","NodeList","State")


def require(ok,code):
    if not ok:
        raise ValueError(code)


@dataclass(frozen=True)
class ExpectedStep:
    parent:str
    step:str
    node:str
    label:str
    intent_SHA256:str

    def __post_init__(self):
        require(re.fullmatch(r"[1-9][0-9]+",self.parent) and re.fullmatch(self.parent+r"\.[0-9]+",self.step)
                and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",self.node)
                and re.fullmatch(r"O01-pack16-[a-z0-9-]+-[0-9a-f]{12}",self.label)
                and re.fullmatch(r"[0-9a-f]{64}",self.intent_SHA256),"ORIGINAL_NUMERIC_STEP_IDENTITY_REQUIRED")


def evidence(scope):
    return {"schema":"O01_PACK16_PORTABLE_STRICT_STEP_EVIDENCE_V1","parent":scope.parent,"step":scope.step,"node":scope.node,
            "intent_SHA256":scope.intent_SHA256,"query":{"class":"NOT_INVOKED","timeout_s":5,"returncode":None,
            "stdout_extent":None,"stdout_SHA256":None,"complete":None,"extent_ok":None,"ascii":None},
            "row_count":None,"fields":{name:{"present":None,"matches":None} for name in FIELDS},
            "TRES":{"gate_reached":False,"present":None,"validator_called":False,"matches":None},"result":"NO_QUERY_RESULT"}


def output_pin(raw,query,complete):
    if type(raw) is bytes:
        query.update(stdout_extent=len(raw),stdout_SHA256=sha256(raw).hexdigest(),complete=complete)


def observe(scope,receipt,query,resource_gate):
    """Preserve the original exact comparison and resource gate return value."""
    meta=receipt["query"]
    try:
        raw=query(["scontrol","show","step",scope.step,"--oneliner"],timeout=5)
    except subprocess.TimeoutExpired as error:
        meta["class"]="TIMEOUT"
        output_pin(error.output,meta,False)
        receipt["result"]="QUERY_TIMEOUT"
        raise
    except subprocess.CalledProcessError as error:
        meta.update({"class":"EXIT_NONZERO","returncode":error.returncode})
        output_pin(error.output,meta,True)
        receipt["result"]="QUERY_NONZERO"
        raise
    except OSError:
        meta["class"]="OS_ERROR"
        receipt["result"]="QUERY_OS_ERROR"
        raise
    except subprocess.SubprocessError:
        meta["class"]="SUBPROCESS_ERROR"
        receipt["result"]="QUERY_SUBPROCESS_ERROR"
        raise
    output_pin(raw,meta,True)
    meta.update({"class":"EXIT_ZERO","returncode":0,"extent_ok":type(raw) is bytes and 0<len(raw)<=16384})
    if not meta["extent_ok"]:
        receipt["result"]="OUTPUT_EMPTY" if raw==b"" else "OUTPUT_OVERSIZE_OR_INVALID"
        raise ValueError("N10_BOUNDED_SLURM_STEP_READ")
    try:
        decoded=raw.decode("ascii")
    except UnicodeError:
        meta["ascii"]=False
        receipt["result"]="OUTPUT_NON_ASCII"
        raise
    meta["ascii"]=True
    rows=[dict(item.split("=",1) for item in line.split() if "=" in item) for line in decoded.splitlines() if line]
    receipt["row_count"]=len(rows)
    receipt["result"]="ROW_COUNT_MISMATCH"
    expected=(scope.step,scope.label,scope.node,"RUNNING")
    if len(rows)==1:
        row=rows[0]
        receipt["result"]="FIELD_MISSING" if any(k not in row for k in FIELDS) else "FIELD_MISMATCH"
        for key,value in zip(FIELDS,expected,strict=True):
            receipt["fields"][key]={"present":key in row,"matches":row.get(key)==value if key in row else None}
    require(len(rows)==1 and tuple(rows[0][key] for key in FIELDS)==expected,"N10_AUTHORITATIVE_NUMERIC_STEP")
    tres=receipt["TRES"]
    receipt["result"]="TRES_MISSING"
    tres.update(gate_reached=True,present="TRES" in rows[0])
    value=rows[0]["TRES"]
    tres["validator_called"]=True
    receipt["result"]="TRES_MISMATCH"
    try:
        resource_gate(value)
    except ValueError:
        tres["matches"]=False
        raise
    tres["matches"]=True
    receipt["result"]="STRICT_NUMERIC_ROW_AND_TRES_PASSED"
    return value


def persist(directory,receipt):
    raw=(json.dumps(receipt,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    require(directory.resolve(strict=True)==directory and directory.is_dir() and len(raw)<=4096,"PRIVATE_BOUNDED_OUTPUT_REQUIRED")
    path=directory/FILENAME
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def query_and_persist(scope,directory,query,resource_gate):
    """Called after physical/allocator scope validation and before Apptainer."""
    result=evidence(scope)
    try:
        return observe(scope,result,query,resource_gate)
    finally:
        persist(directory,result)
