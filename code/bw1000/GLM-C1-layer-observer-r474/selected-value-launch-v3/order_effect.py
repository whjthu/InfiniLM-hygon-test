"""Compare packed and original phases in the same timing slot."""
from statistics import median


def summarize(measured):
    if set(measured) != {"0", "8"}:
        raise ValueError("requires both measured ranks")
    report = {}
    for rank, owner in measured.items():
        for rows in (65, 1313, 2048):
            trials = [r for r in owner["records"] if r["phase"] == "timing" and r["rows"] == rows]
            if [r["packed_first"] for r in trials] != [False, True, True, False] * 4:
                raise ValueError("timing order differs from the published schedule")
            slots = {"first": [], "second": []}
            for i in range(0, len(trials), 2):
                a, b = trials[i:i+2]
                original_first, packed_first = (b, a) if a["packed_first"] else (a, b)
                for slot, packed, original in (
                    ("first", packed_first["packed_ns"], original_first["original_ns"]),
                    ("second", original_first["packed_ns"], packed_first["original_ns"]),
                ):
                    if min(packed, original) <= 0:
                        raise ValueError("nonpositive timing")
                    slots[slot].append(packed/original)
            for slot, samples in slots.items():
                key = f"{rank}/{rows}/{slot}"
                report[key] = dict(ratios=samples, median=median(samples),
                                   faster_blocks=sum(ratio < 1 for ratio in samples))
    reproducible = all(cell["median"] <= .98 and cell["faster_blocks"] >= 7
                       for cell in report.values())
    return dict(cells=report, controlled_component_gain=reproducible,
                decision_rule="All 12 rank/shape/slot cells require median packed/original <= 0.98 and 7/8 individually faster blocks.")
