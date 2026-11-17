"""Independently consume the three original zero-GRES native-input-entry gates."""
import json,hashlib,subprocess
from pathlib import Path
ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
R=ROOT/"training-qwen-treasury-dev3-v2-input-gate-CPU-v1"
COLS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
SPEC={"base":("1087656","a01r4n14"),"old_domain":("1087656","a01r4n16"),"domain":("1103606","f13r1n10")}
artifacts={}
def read(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<1048576
 b=p.read_bytes();artifacts[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(b)
i=read(R/"intent.json");o=read(R/"outcome.json");c=read(R/"source/CLAIM.json");mf=read(R/"source/SOURCE_FILES.json")
assert i["request"]==o["request"]=="136804b0d42744a0ba38f2d84b94a4bf"
assert i["publication"]=="7cd5cd855abca76bc2fd2e632c7456c06143b308"
assert i["epoch"]==o["epoch"] and i["deadline"]==o["deadline"]==i["epoch"]+600
assert o["passed"] and o["all_three_actual_entry_gates_passed"] and o["GPU_GO"] is False
assert artifacts[str(R/"source/CLAIM.json")]=="41e423f95e3fcfec08a12e8737dcd6744ee22870e426f54202d6ccd0a0d66859"
assert artifacts[str(R/"source/SOURCE_FILES.json")]=="8a10db92f02b0b47597da4abf0ea97ebc8e38be90b23b064a8c96124e8e7854d"
assert len(mf["files"])==12
for name,pin in mf["files"].items():
 assert Path(name).name==name
 b=(R/"source"/name).read_bytes();assert len(b)==pin["bytes"] and hashlib.sha256(b).hexdigest()==pin["SHA256"]
frozen=read(R/"source/FROZEN_NATIVE.json")
inventory=read(ROOT/"training-qwen-treasury-explicit-dev-CPU-v1/inventory.json")
assert artifacts[str(ROOT/"training-qwen-treasury-explicit-dev-CPU-v1/inventory.json")]=="07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"
rows=[]
for parent in ("1087656","1103606"):
 text=subprocess.check_output(["sacct","-n","-P","-j",parent,"-o",COLS],text=True,timeout=9)
 rows += [s.split("|") for s in text.splitlines() if R.name+"-" in s]
assert len(rows)==3
arms={}
for arm,(parent,node) in SPEC.items():
 a=R/"arms"/arm;x=read(a/"outcome.json");probe=read(a/"probe.json");step=read(a/"step-intent.json")
 matches=[v for v in rows if v[1]==R.name+"-"+arm]
 assert len(matches)==1 and matches[0]==x["accounting"]["row"]
 assert matches[0][2:4]==["COMPLETED","0:0"] and matches[0][5]==node and "gres/dcu=" not in matches[0][6]
 assert x["passed"] and x["rc"]==0 and x["actual_GPU_cards"]==0
 assert x["probe_SHA256"]==artifacts[str(a/"probe.json")]
 assert probe["passed"] and probe["actual_step"]==matches[0][0] and probe["actual_node"]==node
 assert probe["actual_GPU_cards"]==0 and probe["torch_imported"] is False and probe["explicit_SDK_calls"] is False
 assert probe["generate_called"] is False and probe["tokenize_called"] is False
 assert probe["actual_native_entry"]=="native_generation.bound_inputs"
 assert probe["visual_rows"]==probe["text_only_rows"]==108 and probe["native_source_SHA256"]==frozen
 assert probe["actual_member_SHA256"]==inventory
 assert probe["resolved"]["file_count"]==39 and not probe["resolved"]["missing"] and not probe["resolved"]["extra"] and not probe["resolved"]["linked"]
 assert step["epoch"]==i["epoch"] and step["deadline"]==i["deadline"]
 arms[arm]=dict(actual_accounting=matches[0],probe_SHA256=x["probe_SHA256"],resolved_members=39,
  visual_rows=108,text_only_rows=108,actual_member_SHA256_matches_all39=True,native_source_SHA256=frozen,
  Torch_imported=False,explicit_SDK_calls=False,GPU_cards=0)
print(json.dumps(dict(state="THREE_ORIGINAL_ZERO_GPU_ENTRY_GATES_INDEPENDENTLY_VERIFIED",request=i["request"],
 source_publication=i["publication"],epoch=i["epoch"],deadline=i["deadline"],arms=arms,
 original_artifact_pins=artifacts,originals_passed=True,actual_GPU_steps=0,
 native_data_qualified_for_declared_container_view=True,default_CWD_cover_mechanism_proved=False,
 formal_quality=False,formal_performance=False,automatic_GPU_GO=False)))
