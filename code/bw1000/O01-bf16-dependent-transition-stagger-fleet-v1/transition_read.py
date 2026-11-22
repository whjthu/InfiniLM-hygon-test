"""Plan and independently compare the two fixed physical transition pulse series."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import math

from contract import NANO, ORDER, PROBES, PULSE_OFFSETS, TRANSITION_CENTERS, TRANSITION_SHIFTS
from scope import REQUEST, PARENTS, PCI


SPAN_LIMIT=250_000_000
RT_TOLERANCE=250_000_000
TIMED_CARDS=640


def check(ok, detail):
    if not ok:
        raise ValueError(detail)


def schedule(plan):
    epoch=plan["campaign_epoch"]
    check(plan["request"]==REQUEST and type(epoch) is float and math.isfinite(epoch) and epoch>0,
          "fixed scope or original epoch changed")
    return {phase:dict(request=sha256((REQUEST+":transition:"+phase).encode()).hexdigest()[:32],
                       target_epoch_ns=[round((epoch+offset+relative)*NANO) for relative in PULSE_OFFSETS],
                       expected_card_samples=TIMED_CARDS*len(PULSE_OFFSETS))
            for phase,offset in TRANSITION_CENTERS.items()}


def accounting(pulse, binding):
    rows=pulse["actual_accounting"]
    check(set(rows)==set(PARENTS) and pulse["accounting_qualified"] is True and not pulse["failures"],
          "each observed point requires two independent zero-card terminal sampler steps")
    for parent,row in rows.items():
        check(len(row)==7 and re.fullmatch(parent+r"\.[0-9]+",row[0])
              and row[1]==f"training80-busy-pulse-{binding['request']}-{parent}"
              and row[2:4]==["COMPLETED","0:0"]
              and set(row[6].split(",")) in ({"cpu=40","mem=10G","node=40"},{"cpu=40","mem=10240M","node=40"}),
              "unexpected sampler scope or resource evidence")
    return {parent:row[0] for parent,row in rows.items()}


def read_phase(phase, plan, pulse):
    binding=schedule(plan)[phase]
    check(pulse["request"]==binding["request"] and pulse["complete"] is True
          and pulse["counter"]=="sysfs_gpu_busy_percent" and pulse["GPU_requested"]==0
          and pulse["clock"]=="controller_monotonic_send_to_last_receipt_upper_bound"
          and pulse["planned_nodes"]==80 and pulse["expected_unique_node_PCI"]==pulse["observed_unique_node_PCI"]==640
          and pulse["expected_samples"]==pulse["valid_samples"]==binding["expected_card_samples"]
          and pulse["planned_pulse_interval_ns"]==4*NANO and len(pulse["points"])==5,
          "missing or non-causal transition pulse")
    steps=accounting(pulse,binding)
    nodes=plan["selected"]
    reference=None
    output=[]
    for seq,(point,target) in enumerate(zip(pulse["points"],binding["target_epoch_ns"],strict=True)):
        check(point["sequence"]==seq and point["complete"] is True and point["target_epoch_ns"]==target
              and 0<=point["controller_send_epoch_ns"]-target<=RT_TOLERANCE
              and point["valid_samples"]==640 and not point["missing_nodes"]
              and 0<point["sample_span_upper_bound_ns"]<=SPAN_LIMIT and len(point["worker_reports"])==80,
              "transition point missed the registered epoch, card denominator or causal span")
        by_node={}
        for row in point["worker_reports"]:
            name,parent=row["node"],row["parent"]
            values=row["busy"]
            check(name not in by_node and row["step"]==steps[parent]
                  and row["sequence"]==seq and row["nonce"]==point["nonce"] and row["request"]==binding["request"]
                  and len(values)==8 and all(PCI.fullmatch(pci) for pci in values) and not row["failed_pci"]
                  and all(type(value) is int and 0<=value<=100 for value in values.values()),
                  "invalid node/PCI sample row")
            by_node[name]=(parent,values)
        actual={name:(parent,tuple(sorted(values))) for name,(parent,values) in by_node.items()}
        check(all(sum(parent==entry[0] for entry in actual.values())==40 for parent in PARENTS)
              and (reference is None or reference==actual), "physical fleet membership changed")
        reference=actual
        check(all(by_node[name][0]==item["parent"] and set(by_node[name][1])==set(item["PCI"])
                  for name,item in nodes.items()), "selected operator inventory differs from sampled physical cards")
        selected=[value for name in nodes for value in by_node[name][1].values()]
        fleet=[value for _,values in by_node.values() for value in values.values()]
        output.append(dict(relative_seconds=PULSE_OFFSETS[seq],selected_cards=len(selected),selected_nonzero=sum(v>0 for v in selected),
                           selected_mean_busy=sum(selected)/len(selected),fleet_cards=len(fleet),fleet_nonzero=sum(v>0 for v in fleet),
                           fleet_mean_busy=sum(fleet)/len(fleet),span_upper_bound_ns=point["sample_span_upper_bound_ns"]))
    return reference,output


def summarize(plan, terminal, pulse_by_phase):
    fixed=schedule(plan)
    cards=len(plan["selected"])*8
    check(cards>0 and (terminal["request"],terminal["campaign_epoch"],terminal["planned_nodes"],terminal["planned_cards"])==
          (REQUEST,plan["campaign_epoch"],len(plan["selected"]),cards), "terminal does not bind the same campaign")
    qualified=({cohort % 3 for cohort in plan["cohort_by_shard"].values()}=={0,1,2}
               and terminal["device_qualified"] is True and terminal["valid_cards"]==cards
               and terminal["completed_arms"]==cards*len(ORDER)
               and terminal["actual_busy_card_samples"]==cards*len(ORDER)*len(PROBES)
               and set(terminal["nodes"])==set(plan["selected"]))
    published={}
    issues={}
    memberships={}
    for phase in fixed:
        try:
            memberships[phase],published[phase]=read_phase(phase,plan,pulse_by_phase[phase])
        except (ValueError,KeyError,TypeError) as error:
            issues[phase]=str(error)[:160]
    if not issues and memberships["aligned"]!=memberships["staggered"]:
        issues["physical_membership"]="the two pulse series do not observe identical cards"
    deltas=None
    if qualified and not issues:
        minimum=lambda phase,key:min(row[key] for row in published[phase])
        deltas=dict(worst_selected_nonzero_delta=minimum("staggered","selected_nonzero")-minimum("aligned","selected_nonzero"),
                    worst_selected_mean_busy_point_delta=minimum("staggered","selected_mean_busy")
                    -minimum("aligned","selected_mean_busy"))
    return dict(request=REQUEST,source_schedule=fixed,expected_physical_samples=6400,
                qualified_physical_samples=3200*len(published),fixed_work_qualified=qualified,
                selected_cards=cards,transition_shift_seconds=list(TRANSITION_SHIFTS),points=published,
                issues=issues,paired_sampled_transition_delta=deltas,
                fixed_condition_order=["aligned","staggered"],formal_sustained_busy_qualified=False)


def read(path):
    with path.open("rb") as stream:
        raw=stream.read((16<<20)+1)
    check(0<len(raw)<=16<<20, "unbounded original receipt")
    return json.loads(raw),sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--terminal",type=Path)
    parser.add_argument("--aligned",type=Path)
    parser.add_argument("--staggered",type=Path)
    args=parser.parse_args()
    plan,pin=read(args.plan)
    if args.terminal is None:
        print(json.dumps(schedule(plan),allow_nan=False))
        return 0
    terminal,terminal_pin=read(args.terminal)
    pulses,pins={},{}
    for phase,path in (("aligned",args.aligned),("staggered",args.staggered)):
        if path:
            pulses[phase],pins[phase]=read(path)
    result=summarize(plan,terminal,pulses)
    result.update(plan_SHA256=pin,terminal_SHA256=terminal_pin,pulses_SHA256=pins)
    print(json.dumps(result,allow_nan=False))
    return 0 if result["paired_sampled_transition_delta"] is not None else 2


if __name__=="__main__":
    raise SystemExit(main())
