"""Compare one matched campaign with output identity and fixed arm order."""
from statistics import median

ORDER = ("socket", "shca", "shca", "socket", "socket", "shca")
from collective import ROWS, WIDTH

CASES = tuple((rows, WIDTH) for rows in ROWS)


def compare(arms, modes):
    if tuple(modes) != ORDER or tuple(arm["mode"] for arm in arms) != ORDER:
        raise ValueError("incomplete or incorrect transport arm order")
    results = []
    for index, shape in enumerate(CASES):
        rows = [arm["checks"][index] for arm in arms]
        if any(tuple(row["shape"]) != shape for row in rows):
            raise ValueError("inconsistent collective workloads")
        if len({row["output_SHA256"] for row in rows}) != 1:
            raise ValueError("transport output mismatch")
        ns = [row["median_slowest_rank_ns"] for row in rows]
        if any(t <= 0 for t in ns):
            raise ValueError("invalid measured latency")
        ratios = [ns[1] / ns[0], ns[2] / ns[3], ns[5] / ns[4]]
        results.append(dict(shape=list(shape), dtype="F32", operation="inplace_sum_after_copy", arm_latency_ns=ns,
                            shca_over_socket=ratios, median_shca_over_socket=median(ratios),
                            output_SHA256=rows[0]["output_SHA256"]))
    return results
