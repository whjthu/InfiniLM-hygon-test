"""Verify all original resident prompts and report timing by pair and position."""

import re
from statistics import median

from caller import CHUNKS, ORDERS, PLAN, PROVIDERS, WINDOWS, input_hashes, need, prior


RANKS = set(range(16))
PAIR_NUMBERS = tuple(range(1,len(ORDERS)+1))


def positive(value):
    return type(value) is int and value>0


def progress(record, rank, pci, chunk_sha):
    need(record["rank"]==rank and record["expected_PCI"]==pci
         and record["plans"]==PLAN and record["forwards_planned"]==18,"owner plan identity differs")
    need(len(record["arms"])<=len(PLAN),"excess observed arm records")
    totals={number:dict(attempted=0,returned=0,packed=0) for number in PAIR_NUMBERS}
    last_incomplete=False
    for arm,item in zip(record["arms"],PLAN):
        need(all(arm[k]==v for k,v in item.items()) and not last_incomplete,"owner arm prefix differs")
        rows=arm["fragments"]
        need(len(rows)<=len(CHUNKS),"excess observed forward receipts")
        known=totals[item["pair"]]
        last_incomplete=len(rows)!=len(CHUNKS)
        for offset,(row,(past,count)) in enumerate(zip(rows,CHUNKS)):
            need((row["past"],row["tokens"])==(past,count) and row["IDs_SHA256"]==chunk_sha[past]
                 and row["opt_in"] is (item["arm"]=="ON" and past==0)
                 and row["state"] in {"ENTERED","RETURNED"},"observed original fragment identity differs")
            known["attempted"]+=1
            if row["state"]=="ENTERED":
                need(offset==len(rows)-1,"receipt after failed forward")
                last_incomplete=True
                continue
            known["returned"]+=1
            if "route_counts" in row:
                need(row["opt_in"],"unexpected routes on an unchanged fragment")
                known["packed"]+=prior.counts(row["route_counts"])["packed"]
    need(sum(row["attempted"] for row in totals.values())==record["attempted_forwards"]
         and sum(row["returned"] for row in totals.values())==record["successful_forwards"],
         "attempted or returned denominator differs")
    return totals


def verify_owner(record, rank, pci):
    need(record["PCI"]==pci and record["Native_SHA256"]==prior.NATIVE_SHA
         and record["full_input_SHA256"]==prior.PROMPT_SHA and record["providers"]==PROVIDERS
         and record["state"]=="CLOSED" and record["normal_close"] is True
         and record["attempted_forwards"]==record["successful_forwards"]==18
         and record["reset_calls"]==record["device_input_audits"]==7 and len(record["arms"])==len(PLAN)
         and record["Model_GO"] is record["model_speed_win"] is False,
         "owner did not close the full six-arm experiment")
    load=record["load_progress"]
    need(record["context_creations"]==record["checkpoint_loads"]==load["context_creations"]
         ==load["context_create_attempts"]==load["checkpoint_loads"]==load["checkpoint_load_attempts"]==1
         and record["parameters"]==load["checkpoint_key_count"]==load["loaded_parameters"]==prior.LOAD_KEYS
         and record["layers"]==prior.LAYERS and load["actual_layers"]==list(range(prior.LAYERS))
         and load["all78layers_loaded"] is True, "actual one-load checkpoint evidence differs")
    driver=prior.driver_summary(lambda:record["driver_memory"])
    memory=[record["final_memory"],record["memory_after_close"]]
    pairs={number:dict(packed=0,bitwise_fragments=0 if rank==0 else None) for number in PAIR_NUMBERS}
    clock_rows=[]
    first={}
    off_outputs=[]
    for arm in record["arms"]:
        memory.extend((arm["memory_before"],arm["memory_after"]))
        if arm["position"]==1:
            first={}
        counts=pairs[arm["pair"]]
        times=[]
        for row in arm["fragments"]:
            need(row["state"]=="RETURNED" and positive(row["forward_ns"]),"actual bounded forward clock missing")
            times.append(row["forward_ns"])
            memory.append(row["memory_after"])
            if row["opt_in"]:
                packed=prior.counts(row["observed_route_counts"])
                need(row["route_counts"]==packed,"first-fragment route counts differ")
                counts["packed"]+=packed["packed"]
            else:
                need("route_counts" not in row,"unexpected routes after first selected fragment")
            digest=row.get("root_BF16_logits_SHA256")
            if rank==0:
                need(isinstance(digest,str) and re.fullmatch(r"[0-9a-f]{64}",digest),"complete rank0 BF16 digest missing")
                if arm["position"]==1:
                    first[row["past"]]=digest
                    need("bitwise_equal_to_pair_first" not in row,"first arm cannot claim a pair comparison")
                else:
                    equal=(digest==first[row["past"]])
                    need(row["bitwise_equal_to_pair_first"] is equal,"pair byte result and output digest disagree")
                    counts["bitwise_fragments"]+=int(equal)
            else:
                need(digest is None and "bitwise_equal_to_pair_first" not in row,"non-root claimed root logits")
        need(len(times)==len(CHUNKS) and arm["first_forward_ns"]==times[0]
             and arm["sum_three_forward_ns"]==sum(times) and type(arm["host_scope_ns"]) is int
             and arm["host_scope_ns"]>=sum(times),"actual first, CABI-sum or host interval differs")
        clock_rows.append({k:arm[k] for k in ("pair","position","arm",*WINDOWS)})
        if rank==0 and arm["arm"]=="OFF":
            off_outputs.append(tuple(row["root_BF16_logits_SHA256"] for row in arm["fragments"]))
    need(len({r["total"] for r in memory})==1
         and all(type(r["free"]) is type(r["total"]) is int and prior.FLOOR<=r["free"]<=r["total"] for r in memory),
         "original 8GiB boundary evidence differs")
    return dict(rank=rank,pairs=pairs,minimum_boundary_free=min(r["free"] for r in memory),
                minimum_sampled_free=driver["min_observed"],arms=clock_rows,
                rank0_OFF_outputs_stable=(len(set(off_outputs))==1 if rank==0 else None))


def timing_report(verified):
    pairs=[]
    positions={1:{"OFF":[],"ON":[]},2:{"OFF":[],"ON":[]}}
    for p,order in enumerate(ORDERS,1):
        arm_rows=[]
        for place,arm in enumerate(order,1):
            rows=[item["arms"][(p-1)*2+place-1] for item in verified]
            slowest={}
            for name in WINDOWS:
                rank,item=max(zip(verified,rows),key=lambda ri:ri[1][name])
                slowest[name]={"ns":item[name],"rank":rank["rank"]}
            arm_rows.append(dict(position=place,arm=arm,slowest_rank=slowest))
            positions[place][arm].append(dict(pair=p,slowest_rank=slowest))
        off,on=sorted(arm_rows,key=lambda row:row["arm"])
        ratio={k:on["slowest_rank"][k]["ns"]/off["slowest_rank"][k]["ns"] for k in WINDOWS}
        pairs.append(dict(pair=p,arms=arm_rows,ON_over_OFF=ratio))
    return dict(pairs=pairs,positions=positions,median_ON_over_OFF={k:median(row["ON_over_OFF"][k] for row in pairs)
                                                                  for k in WINDOWS})


def summarize(records, admitted_pci, token_ids):
    need(set(admitted_pci)==RANKS and all(re.fullmatch(prior.PCI_PATTERN,p) for p in admitted_pci.values())
         and all(len({admitted_pci[r] for r in node})==8 for node in (range(8),range(8,16))),
         "expected eight fresh physical PCI functions per node")
    need(set(records)<=RANKS,"unexpected rank in primary results")
    chunk_sha=input_hashes(token_ids)
    accepted=[]
    rejected=[]
    ledger=[]
    totals={p:dict(pair=p,forwards_planned=96,attempted=0,returned=0,route_layers_planned=16*prior.LAYERS,packed=0)
            for p in PAIR_NUMBERS}
    for rank in sorted(RANKS):
        row=records.get(rank)
        partial=None
        if row is None:
            rejected.append(dict(rank=rank,reason="MISSING"))
        else:
            try:
                partial=progress(row,rank,admitted_pci[rank],chunk_sha)
                for p,observed in partial.items():
                    for key,value in observed.items():
                        totals[p][key]+=value
                accepted.append(verify_owner(row,rank,admitted_pci[rank]))
            except (IndexError,KeyError,TypeError,ValueError) as error:
                rejected.append(dict(rank=rank,reason=str(error)[:256]))
        ledger.append(dict(rank=rank,owner_state=row.get("state","UNKNOWN") if type(row) is dict else "MISSING",
                           forwards_planned=18,observed_prefix=partial))
    root=next((row for row in accepted if row["rank"]==0),None)
    for number,row in totals.items():
        row["rank0_bitwise_fragments_planned"]=3
        row["rank0_bitwise_fragments"]=root["pairs"][number]["bitwise_fragments"] if root else None
    stable=root["rank0_OFF_outputs_stable"] if root else None
    qualified=(len(accepted)==len(RANKS) and stable is True and all(row["packed"]==row["route_layers_planned"]
               and row["rank0_bitwise_fragments"]==3 for row in totals.values()))
    return dict(owners_planned=16,owners_verified=len(accepted),pairs_planned=len(ORDERS),prompts_planned=6,
                owner_forwards_planned=288,route_layers_planned=3744,root_bitwise_fragments_planned=9,
                rejected=rejected,owner_progress=ledger,pairs=list(totals.values()),
                rank0_OFF_outputs_stable=stable,prefill_numerical_evidence_pass=qualified,
                timing_evidence=timing_report(accepted) if qualified else None,
                GPU_GO=False,Model_GO=False,model_speed_win=False,decode_qualified=False)
