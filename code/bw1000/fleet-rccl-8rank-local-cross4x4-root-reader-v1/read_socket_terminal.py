import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/'fleet-rccl-8rank-local-cross4x4-v1-b6fea3ae8c774a6da17cf41acfb0f6f9'
REQUEST = 'b6fea3ae8c774a6da17cf41acfb0f6f9'

def read(rel):
    path=ROOT/rel
    total=path.stat().st_size
    if not 0<total<2*1024**2:
        raise ValueError("BOUNDED_RECEIPT")
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def account():
    matches={}
    for j in ("1087656","1103606"):
        out=subprocess.check_output(['sacct','-n','-P','-j',j,
            '--format=JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'],text=True,timeout=9)
        matches[j]=[v.split('|') for v in out.splitlines() if 'rccl-local-cross4x4' in v and REQUEST in v]
    return matches

def sample(available, epoch):
    verification=collections.defaultdict(list)
    for n in available:
        for f in (ROOT/'nodes'/n).glob('busy-[0-9]*.json'):
            d=json.loads(f.read_bytes())
            if d["node"]!=n or d["step"]!=available[n]["step"]:
                raise ValueError("BUSY_SAME_STEP")
            start=epoch+int(f.stem.split("-")[1])
            values=[]
            for i,row in enumerate(d["rows"]):
                v=row["busy"]
                if (row["point"]!=i or not start+2*i <=row["opened_epoch"]<=row["closed_epoch"]<=start+2*i+1.0
                        or set(v)!=set(available[n]["pci"])
                        or any(type(x) is not int or not 0<=x<=100 for x in v.values())):
                    raise ValueError("BUSY_PCI_AND_WINDOW")
                values.extend(v.values())
            if len(values)!=24:
                raise ValueError("BUSY_8_DEVICES_TIMES_3")
            verification[f.stem].extend(values)
    return {k:dict(devices=len(v)//3,samples=len(v),mean=sum(v)/len(v)) for k,v in verification.items()}

def required(condition,code):
    if not condition:
        raise ValueError(code)

def main():
    accounts=account()
    if not (ROOT/'dispatch-terminal.json').is_file():
        return dict(phase="ACTUAL_DISPATCH_NOT_TERMINAL",slurm=accounts)
    intent,intent_sha=read('intent.json')
    original,original_sha=read('dispatch-terminal.json')
    required(intent['request']==original['request']==REQUEST,"REQUEST")
    cohort=intent['pair_by_node']
    expected={n:j for j,ns in intent['parents'].items() for n in ns}
    required(set(cohort)==set(expected) and len(expected)==75,"NODE_DENOMINATOR")
    observed={}; failed={}; pins={}; transport=collections.Counter()
    for node,j in sorted(expected.items()):
        base=f"nodes/{node}"
        try:
            terminal,pins[base+"/terminal.json"]=read(base+"/terminal.json")
            inventory,pins[base+"/inventory.json"]=read(base+"/inventory.json")
            step=terminal["step"]
            required(any(r[0]==step for r in accounts[j]),"EXACT_SLURM_STEP")
            required(not terminal["error"] and terminal["post_kfd_holders"]==0 and
                      terminal["exits"]=={str(i):0 for i in range(8)},"WORKER_TERMINAL_OR_KFD")
            required(inventory["cohort"]==cohort[node],"PER_NODE_TOPOLOGY")
            cards={}
            for i in range(8):
                name=f"{base}/card-{i}.json"
                card,pins[name]=read(name)
                report,=card["results"]
                required(card["card"]==i and card["cohort"]==cohort[node] and card["step"]==step
                         and card["error"] is None,"CARD_IDENTITY")
                blocks=report["blocks"]
                required([b["arm"] for b in blocks]==list(range(4)),"ABBA_ORDER")
                for k,b in enumerate(blocks):
                    expected_cross=len(cohort[node])==2 and k in (1,2)
                    rank=cohort[node].index(node)*4+i%4 if expected_cross else i
                    required(b["cross"] is expected_cross and b["logical_rank"]==rank and
                             b["payload_bytes"]==26214400 and b["verified_elements"]==13107200 and
                             b["payload_dtype"]=="float32" and b["measured_reductions"]==192 and
                             len(b["rank_max_seconds"])==192 and all(math.isfinite(x) and x>0 for x in b["rank_max_seconds"]),
                             "BLOCK_WORK_AND_RESULT")
                log=ROOT/base/f'card-{i}.log'
                if log.is_file() and log.stat().st_size<512*1024:
                    markers=set(re.findall(rb'Using network (Socket|IB)\b', log.read_bytes(), re.I))
                    transport.update(v.decode('ascii').upper() for v in markers)
                cards[i]=dict(blocks=blocks,pci=card["pci"])
            pci={row["pci"] for row in cards.values()}
            required(pci==set(inventory["PCI_KFD"]) and len(pci)==8,"PHYSICAL_CARDS")
            for arm in (0,3):
                required(all(cards[i]["blocks"][arm]["rank_max_seconds"]==cards[0]["blocks"][arm]["rank_max_seconds"]
                             for i in range(8)),"LOCAL_RANK_MAX_AGREEMENT")
            observed[node]=dict(cards=cards,pci=pci,step=step)
        except (OSError,ValueError,KeyError,TypeError) as fault:
            failed[node]=str(fault) if isinstance(fault,ValueError) and re.fullmatch("[A-Z_0-9]+",str(fault)) else type(fault).__name__
    node_pairs=sorted({tuple(v) for v in cohort.values() if len(v)==2})
    pair_rows=[]; pair_failures={}
    for a,b in node_pairs:
        if a not in observed or b not in observed:
            pair_failures[a+"+"+b]="MISSING_MEMBER"
            continue
        arrays=[observed[n]["cards"] for n in (a,b)]
        local=[]; cross=[]
        for cards in arrays:
            local.extend(t for arm in (0,3) for t in cards[0]["blocks"][arm]["rank_max_seconds"])
        for half in range(2):
            for arm in (1,2):
                values=arrays[0][half*4]["blocks"][arm]["rank_max_seconds"]
                if not all(c[i]["blocks"][arm]["rank_max_seconds"]==values for c in arrays for i in range(half*4,half*4+4)):
                    pair_failures[a+"+"+b]="CROSS_RANK_MAX_AGREEMENT"
                cross.extend(values)
        if a+"+"+b not in pair_failures:
            x,y=statistics.median(local),statistics.median(cross)
            pair_rows.append(dict(nodes=[a,b],parent=expected[a],local_ms=1000*x,cross_ms=1000*y,cross_over_local=y/x))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),slurm=accounts,phase="ACTUAL_DISPATCH_TERMINAL",
        terminal_SHA256=original_sha,intent_SHA256=intent_sha,observed_cards=8*len(observed),
        planned_pairs=len(node_pairs),complete_pairs=len(pair_rows),cross_cards=len(pair_rows)*16,
        independent_local_control_cards=sum(len(cohort[n])==1 for n in observed)*8,
        pair_failures=pair_failures,node_failures=failed,per_pair=pair_rows,transport_log_process_markers=dict(transport),
        worker_receipts_SHA256=hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest(),
        busy=sample(observed,intent["sample_epoch"]))
    if pair_rows:
        result["median"]= {k:statistics.median(r[k] for r in pair_rows) for k in ("local_ms","cross_ms","cross_over_local")}
    return result

print(json.dumps(main(),allow_nan=False))
