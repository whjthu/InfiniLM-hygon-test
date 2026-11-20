"""Reduce each measured sample over all physical ranks."""
from statistics import median
from collective import ORDER, ROWS, WIDTH


def compare(ranks):
    if [record["rank"] for record in ranks] != list(range(16)):
        raise ValueError("incorrect rank denominator")
    cases = [(row, variant) for row in ROWS for variant in ORDER]
    if any(len(record["checks"]) != len(cases) for record in ranks):
        raise ValueError("incomplete pair denominator")
    by_rows = {}
    for index, (row, variant) in enumerate(cases):
        values = [record["checks"][index] for record in ranks]
        for rank, value in enumerate(values):
            if (value["shape"] != [row,WIDTH] or value["variant"] != variant or value["dtype"] != "F32"
                    or value["ranks"] != 16 or value["rank"] != rank or value["bytes_per_rank"] != row*WIDTH*4
                    or value["guard_bytes_each_side"] != 256 or value["checked_iterations"] != 25
                    or len(value["samples"]) != 20
                    or any(type(s["wall_ns"]) is not int or s["wall_ns"] <= 0 for s in value["samples"])):
                raise ValueError("invalid F32 paired result")
        hashes = {v["output_SHA256"] for v in values}
        if len(hashes) != 1:
            raise ValueError("F32 output disagreement between ranks")
        block = dict(variant=variant, output_SHA256=hashes.pop())
        for clock in ("wall_ns","submit_ns","sum_stream_ms","copy_stream_ms"):
            block["median_slowest_rank_"+clock] = median(max(samples) for samples in
                zip(*(tuple(s[clock] for s in v["samples"]) for v in values)))
        by_rows.setdefault(row,[]).append(block)
    return {row: blocks[0] for row, blocks in by_rows.items()}




def compare_transports(arms, order):
    if order != ("socket", "shca", "shca", "socket", "socket", "shca"):
        raise ValueError("incorrect TP16 transport pair order")
    if len(arms) != len(order) or any(set(arm) != set(ROWS) for arm in arms):
        raise ValueError("incomplete transport pair denominator")
    result = []
    for row in ROWS:
        blocks = [dict(transport=mode, **arm[row]) for mode, arm in zip(order, arms)]
        if len({block["output_SHA256"] for block in blocks}) != 1:
            raise ValueError("F32 sum mismatch between independent communicators")
        times = [block["median_slowest_rank_wall_ns"] for block in blocks]
        ratios = [times[1]/times[0], times[2]/times[3], times[5]/times[4]]
        result.append(dict(shape=[row,WIDTH], blocks=blocks, shca_over_socket=ratios,
                           paired_median_ratio=median(ratios)))
    return result
