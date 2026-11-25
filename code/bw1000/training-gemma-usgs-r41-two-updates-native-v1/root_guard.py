"""Fresh exact target control-plane and old-holder proof before a held launch."""

from datetime import datetime,timezone
import json
import os
import re
import time

from contract import D, LABEL, NODE, PARENT, R, REQUEST, WHOLE, stable_json
from prepare import read, need


OLD_DATA_REQUEST='6d14e9df9832432b97f1f46f032b49f3'
OLD_OUTCOME_SHA='cf2315232e6dbe68bdf1ae83943bc9a7b54c49a12731cea44ce4f443ea954d5f'
OLD_NATIVE='training-gemma-usgs-r39-two-updates-native-v1'
OLD_STEPS={'pre': ['1103606.3416', 'training-gemma-usgs-r39-two-updates-native-v1-pre', 'COMPLETED', '0:0', '0', 'f13r1n12', 'cpu=2,mem=4G,node=1'], 'gpu': ['1103606.3417', 'training-gemma-usgs-r39-two-updates-native-v1-gpu', 'FAILED', '1:0', '1', 'f13r1n12', 'cpu=64,gres/dcu=8,mem=224000M,node=1'], 'post': ['1103606.3418', 'training-gemma-usgs-r39-two-updates-native-v1-post', 'COMPLETED', '0:0', '0', 'f13r1n12', 'cpu=2,mem=4G,node=1']}
OLD_STEP="1103606.3355"
OLD_LABEL="training80-dcu-standby-4d63273bbce5-f13r1n12"
STABLE_TERMINAL={"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL","DEADLINE","REVOKED"}


def fields(line):
    return dict(atom.split("=",1) for atom in line.split() if "=" in atom)


def check_authority(approval,now):
    need(approval["request_id"]==REQUEST and approval["node"]==NODE and approval["parent"]==PARENT
         and approval["approved_eight_card_single_attempt"] is approval["ordinary_exclusion"] is True
         and approval["release_requires_terminal_independent_post"] is True and approval["old_holder_step"]==OLD_STEP
         and approval["coordinator_identity_verified"] is True and approval["manual_adoption_authorized"] is True
         and approval["prior_native_accounting_verified"] is True
         and approval["cpu_original_independent_terminal_SHA256"]=='74bac702f6e94de63d3db93ad8744fffdb45b797881510792587ef76e415de02'
         and approval["valid_until"]>=now+WHOLE and 0<=now-approval["issued_epoch"]<=90,
         "R39_FRESH_SOLE_OWNER_AND_EXCLUSION_AUTHORITY")


def verify_prior_native(rows):
    records = {name: [row for row in rows if row[0] == expected[0]] for name, expected in OLD_STEPS.items()}
    need(all(records[name] == [expected] for name, expected in OLD_STEPS.items()),
         "R41_EXACT_ORIGINAL_3416_3417_3418_TERMINALS")
    root = R/"results"/OLD_NATIVE
    old, pin = stable_json(root/"outcome.json")
    need(pin["SHA256"]==OLD_OUTCOME_SHA and old["request_id"]==OLD_DATA_REQUEST
         and old["release_permitted"] is True and old["engineering_completed"] is False,
         "R41_EXACT_ORIGINAL_3417_CLOSED_OWNER")
    before,_ = stable_json(root/"before.json",old["before"])
    after,_ = stable_json(root/"after.json",old["after"])
    need(before["step"]==OLD_STEPS["pre"][0] and after["step"]==OLD_STEPS["post"][0]
         and before["boot_id"]==after["boot_id"] and before["pci"]==after["pci"]
         and len(set(after["pci"]))==8 and before["kfd_holder_count"]==after["kfd_holder_count"]==0
         and after["owned_processes_remaining"]==[], "R41_EXACT_ORIGINAL_3418_INDEPENDENT_RELEASE")


def admit_r41(query,rows,clock):
    claim=json.loads((D/"CLAIM.json").read_bytes())
    raw=read(R/"incoming"/(LABEL+"-authority")/(REQUEST+".json"),claim["placement_authority_SHA256"],16384)
    approval=json.loads(raw)
    current=time.time()
    check_authority(approval,current)
    accounting=rows(clock)
    verify_prior_native(accounting)
    observed=[row for row in accounting if row[0]==OLD_STEP]
    need(len(observed)==1 and observed[0]==approval["old_holder_accounting"] and observed[0][1]==OLD_LABEL
         and observed[0][2:4]==["COMPLETED","0:0"] and observed[0][5]==NODE
         and "gres/dcu=8" in observed[0][6].split(","),"R39_REAL_ORIGINAL_3355_C0_FIRST")
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
