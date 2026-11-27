"""Fresh exact target control-plane and old-holder proof before a held launch."""

from datetime import datetime,timezone
import json
import os
import re
import time

from contract import D, LABEL, NODE, PARENT, R, REQUEST, WHOLE, allocated, stable_json
from prepare import read, need


OLD_DATA_REQUEST='9f59101518e6424b98fa3472a0323f44'
OLD_OUTCOME_SHA='00d435e47c82bfae723e9a67fa61b00230ca0baa8a332e5d3e1c50248b8142a5'
OLD_NATIVE='training-gemma-usgs-r44-live-sampler-diagnostic-v1'
OLD_STEPS={'gpu': ['1103606.3465', 'training-gemma-usgs-r44-live-sampler-diagnostic-v1-gpu', 'FAILED', '1:0', '104', 'f13r1n12', 'cpu=64,gres/dcu=8,mem=224000M,node=1'], 'post': ['1103606.3466', 'training-gemma-usgs-r44-live-sampler-diagnostic-v1-post', 'COMPLETED', '0:0', '1', 'f13r1n12', 'cpu=2,mem=4G,node=1'], 'pre': ['1103606.3464', 'training-gemma-usgs-r44-live-sampler-diagnostic-v1-pre', 'COMPLETED', '0:0', '1', 'f13r1n12', 'cpu=2,mem=4G,node=1']}
CPU3481=['1103606.3481', 'training-gemma-r45-swift-sampler-cpu-projection-v1', 'COMPLETED', '0:0', '6', 'f13r1n12', 'cpu=4,mem=8G,node=1']
CPU3481_WITNESS='b6e32e773064800a697ec3dd6641916cac99e50f78634c0fcaf117cabf174d1e'
SAMPLER_SOURCE='1766b92110bdc702e7a26716e2574d128d06b76ada65871b3c53082e85d43d5b'
PRIOR_RELEASE_SHA='e7bf4c1417c7094536b599c0c19b26793a57ace14fce2ab8d01079ffdeda2e19'
HOLDER_ROW=['1103606.3465', 'training-gemma-usgs-r44-live-sampler-diagnostic-v1-gpu', 'FAILED', '1:0', '104', 'f13r1n12', 'cpu=64,gres/dcu=8,mem=224000M,node=1', '2026-11-25T13:27:28', '2026-11-25T13:29:12']
HOLDER_POST=['1103606.3466', 'training-gemma-usgs-r44-live-sampler-diagnostic-v1-post', 'COMPLETED', '0:0', '1', 'f13r1n12', 'cpu=2,mem=4G,node=1', '2026-11-25T13:29:15', '2026-11-25T13:29:16']
HOLDER_OBSERVATION_SHA='a98822cd96e2679b7223935d18fe115046204ec2f99e0446d9368dc353b8c0fb'
STABLE_TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}


def fields(line):
    return dict(atom.split("=",1) for atom in line.split() if "=" in atom)


def check_authority(approval,now):
    need(approval["request_id"]==REQUEST and approval["node"]==NODE and approval["parent"]==PARENT
         and approval["approved_eight_card_single_attempt"] is approval["ordinary_exclusion"] is True
         and approval["release_requires_terminal_independent_post"] is True
         and approval["latest_explicit_holder_verified_terminal"] is True
         and approval["cpu_3481_independent_SHA256"]==CPU3481_WITNESS
         and approval["sampler_source_manifest_SHA256"]==SAMPLER_SOURCE
         and approval["prior_diagnostic_terminal_SHA256"]==PRIOR_RELEASE_SHA
         and approval["coordinator_identity_verified"] is True and approval["manual_adoption_authorized"] is True
         and approval["prior_native_accounting_verified"] is True
         and approval["cpu_original_independent_terminal_SHA256"]=='74bac702f6e94de63d3db93ad8744fffdb45b797881510792587ef76e415de02'
         and approval["valid_until"]>=now+WHOLE and 0<=now-approval["issued_epoch"]<=90,
         "R39_FRESH_SOLE_OWNER_AND_EXCLUSION_AUTHORITY")


def verify_prior_native(rows):
    records = {name: [row for row in rows if row[0] == expected[0]] for name, expected in OLD_STEPS.items()}
    need(all(records[name] == [expected] for name, expected in OLD_STEPS.items()),
         "R46_EXACT_ORIGINAL_3464_3465_3466_TERMINALS")
    root = R/"results"/OLD_NATIVE
    old, pin = stable_json(root/"outcome.json")
    need(pin["SHA256"]==OLD_OUTCOME_SHA and old["request_id"]==OLD_DATA_REQUEST
         and old["release_permitted"] is True and old["engineering_completed"] is False,
         "R46_EXACT_ORIGINAL_3465_CLOSED_OWNER")
    before,_ = stable_json(root/"before.json",old["before"])
    after,_ = stable_json(root/"after.json",old["after"])
    need(before["step"]==OLD_STEPS["pre"][0] and after["step"]==OLD_STEPS["post"][0]
         and before["boot_id"]==after["boot_id"] and before["pci"]==after["pci"]
         and len(set(after["pci"]))==8 and before["kfd_holder_count"]==after["kfd_holder_count"]==0
         and after["owned_processes_remaining"]==[], "R46_EXACT_ORIGINAL_3466_INDEPENDENT_RELEASE")


def verify_latest_holder(approval, accounting, hosts, now):
    """Accept only the independently released original .3465 and veto later conflicts."""
    need(approval["old_holder_step"]==HOLDER_ROW[0] and approval["latest_holder_accounting"]==HOLDER_ROW
         and approval["latest_holder_observation_SHA256"]==HOLDER_OBSERVATION_SHA,
         "R46_EXACT_3465_FRESH_AUTHORITY")
    expected={row[0]:row for row in (HOLDER_ROW,HOLDER_POST)}
    for handle, row in expected.items():
        need([found for found in accounting if found[0]==handle]==[row],
             "R46_EXACT_3465_AND_3466_NINE_COLUMN_ACCOUNTING")
    def timestamp(value):
        return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    def interval(row):
        need(re.fullmatch(r"[0-9]{1,8}",row[4]) is not None,"R46_ACCOUNTING_FINITE_ELAPSED")
        start,end=map(timestamp,row[7:9])
        need(0<start<=end<=now+1 and abs(end-start-int(row[4]))<=1,"R46_ACCOUNTING_FINITE_START_END")
        return start,end
    start,end=interval(HOLDER_ROW)
    post_start,_=interval(HOLDER_POST)
    need(start<=end<=post_start,"R46_3465_TERMINAL_BEFORE_3466_RELEASE")
    number=int(HOLDER_ROW[0].rsplit(".",1)[1])
    seen=set()
    for row in accounting:
        handle=row[0]
        if not re.fullmatch(PARENT+r"\.[0-9]+",handle) or int(handle.rsplit(".",1)[1])<=number:
            continue
        dcu=allocated(row[6]) if row[6] else None
        if row[5] in ("","(null)","N/A","Unknown","None assigned"):
            need(dcu==0 and row[2] in STABLE_TERMINAL,"R46_LATER_STEP_UNCLASSIFIED_PLACEMENT")
            continue
        if NODE not in hosts(row[5]):
            continue
        need(handle not in seen,"R46_LATER_STEP_ID_AMBIGUOUS")
        seen.add(handle)
        need(dcu==0 and row[2] in STABLE_TERMINAL,"R46_LATER_N12_GPU_OR_UNKNOWN_OWNER")
        interval(row)


def holder_accounting(query, clock):
    from utc_scope import history_argv
    text=query(history_argv(PARENT),clock)
    need(0<len(text)<=2<<20,"R46_HOLDER_ACCOUNTING_EXTENT")
    rows=[row.split("|") for row in text.splitlines() if row]
    need(len(rows)<=8192 and all(len(row)==9 for row in rows),"R46_HOLDER_ACCOUNTING_NINE_COLUMNS")
    return rows


def admit_r41(query,rows,clock):
    from r120_holder import verify_latest_r120
    verify_latest_r120(query,clock)
    claim=json.loads((D/"CLAIM.json").read_bytes())
    raw=read(R/"incoming"/(LABEL+"-authority")/(REQUEST+".json"),claim["placement_authority_SHA256"],16384)
    approval=json.loads(raw)
    current=time.time()
    check_authority(approval,current)
    need(approval["latest_release_SHA256"]==claim["latest_holder_observation_SHA256"],
         "R119_ROOT_AUTHORITY_RELEASE_BINDING")
    accounting=rows(clock)
    verify_prior_native(accounting)
    need([row for row in accounting if row[0]==CPU3481[0]]==[CPU3481],"R46_ORIGINAL_3481_ZERO_GRES_ACCOUNTING")
    proof=json.loads(read(D/"IPC_4251.json","13e36e282125142e9bdbe15675b192159e99db15511a27e2be0d446cbf44dd54",16384))
    evidence=proof["independent"]
    actual=holder_accounting(query,clock)
    need(approval["ipc_zero_gres_4251_SHA256"]==claim["ipc_zero_gres_4251_SHA256"]=="13e36e282125142e9bdbe15675b192159e99db15511a27e2be0d446cbf44dd54"
         and [row for row in actual if row[0]=="1103606.4251"]==[proof["accounting"]]
         and proof["accounting"][2:7]==["COMPLETED","0:0","5",NODE,"cpu=4,mem=8G,node=1"]
         and evidence["passed"] is evidence["marker_only_natural"] is True
         and evidence["registered_rmdirs"]==evidence["completed_rows"]==2
         and evidence["mapped_provider_source_files"]==6,
         "R120_EXACT_REAL_4251_DUAL_PROVIDER_RETIREMENT")
    def hosts(spec):
        if re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",spec):
            return {spec}
        need(spec not in ("","(null)","N/A","Unknown"),"R39_EXACT_NODE_MEMBERSHIP")
        return set(query(["scontrol","show","hostnames",spec],clock).splitlines())
    output=query(["scontrol","show","job",PARENT,"-o"],clock).splitlines()
    need(len(output)==1,"R39_EXACT_CONTROLLER_PARENT_ROW")
    parent=fields(output[0])
    need(parent["JobId"]==PARENT and parent["JobState"]=="RUNNING" and parent["UserId"].endswith(f"({os.getuid()})")
         and NODE in hosts(parent["NodeList"]) and "gres/dcu=320" in parent["AllocTRES"].split(",")
         and datetime.fromisoformat(parent["EndTime"]).replace(tzinfo=timezone.utc).timestamp()>=current+WHOLE,
         "R39_ACTUAL_PARENT_AND_FULL_EPOCH")
    names=[]
    for line in query(["scontrol","-d","show","job",PARENT],clock).splitlines():
        for nodes,cpus,mem in re.findall(r"\bNodes=(\S+)\s+CPU_IDs=(\S+)\s+Mem=(\S+)",line):
            if NODE in hosts(nodes):
                names.append((cpus,mem))
    need(len(names)==1,"R39_PER_NODE_PARENT_CPU_MEMORY_UNAVAILABLE")
    cpu,mem=names[0]
    slots=set()
    for item in cpu.split(","):
        match=re.fullmatch(r"([0-9]+)(?:-([0-9]+))?",item)
        need(match is not None,"R39_CONTROLLER_CPU_SLOT_SYNTAX")
        first,last=int(match[1]),int(match[2] or match[1])
        need(0<=first<=last<=16384,"R39_CONTROLLER_CPU_SLOT_BOUNDS")
        slots.update(range(first,last+1))
    match=re.fullmatch(r"([0-9]+)([MGT]?)",mem)
    need(match is not None and len(slots)>=64 and int(match[1])*{"":1,"M":1,"G":1024,"T":1048576}[match[2]]>=224000,
         "R39_FULL_PARENT_CPU_AND_MEMORY")
    for line in query(["scontrol","show","step",PARENT,"-o"],clock).splitlines():
        step=fields(line)
        ident=step["StepId"]
        need(re.fullmatch(PARENT+r"\.(?:[0-9]+|batch|extern)",ident),"R39_EXACT_CONTROLLER_STEP_MEMBERSHIP")
        if ident.rsplit(".",1)[1].isdecimal() and step["State"].rstrip("+") not in STABLE_TERMINAL:
            need(NODE not in hosts(step["NodeList"]),"R39_UNEXPECTED_LIVE_NUMBERED_STEP_CONTROLLER")
