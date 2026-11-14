import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = "results/bw1000/2026-11-13/"
SOURCES = {
    "socket": (ROOT + "added-o01-bf16-collective-1103606-4.json", "rank_records", "1103606.4"),
    "shca": (ROOT + "added-o01-shca-collective-1103606-24.json", "rank_results", "1103606.24"),
}
GLM = ROOT + "GLM474-TP16-native-collective-1103606-66-terminal.json"


def read(root, path):
    data = (root / path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def window(rows):
    assert len(rows) == 2 and all(len(row) == 20 for row in rows)
    assert all(type(value) is int and value > 0 for row in rows for value in row)
    ordered = sorted(max(first, second) for first, second in zip(*rows))
    return dict(measured_windows=len(ordered), median_us=statistics.median(ordered) / 1000,
                p95_us=ordered[math.ceil(.95 * len(ordered)) - 1] / 1000,
                max_us=ordered[-1] / 1000)


def audit(root):
    pins = {}
    identities = {}
    runs = {}
    for arm, (path, key, handle) in SOURCES.items():
        doc, pins[path] = read(root, path)
        independent = doc["independent"]
        assert independent["passed"] and independent["verified"]
        accounting = independent["rows"]
        assert len(accounting) == 3 and accounting[1][0] == handle
        assert all(row[2:4] == ["COMPLETED", "0:0"] for row in accounting)
        if arm == "socket":
            assert doc["pre_and_post_clean"] and doc["selected_channel_transport"] == "Socket/1"
        else:
            assert doc["clean_post"] and set(independent["network"]) == {"IBext_v8"}
            assert independent["plugin_channels"] > 0 and independent["socket_channels"] == 0
        ranks = sorted(doc[key], key=lambda rank: rank["rank"])
        assert [rank["rank"] for rank in ranks] == [0, 1]
        assert [rank["node"] for rank in ranks] == ["f12r2n10", "f12r2n11"]
        identities[arm] = [(r["node"], r["pci"], r["kfd"]["gpu_id"], r["boot_id"],
                            r["GPU_name"], r["HIP"], r["Torch"], r["backend"]) for r in ranks]
        results = {}
        for index, size in enumerate((8192, 131072, 4194304)):
            samples = [rank["rows"][index] for rank in ranks]
            assert all(s["bytes"] == size == s["elements"] * 2 and s["checked"] == 25
                       for s in samples)
            result = window([s["latency_ns"] for s in samples])
            assert independent["summary"][index] == dict(bytes=size, median_slowest_rank_us=result["median_us"])
            results[size] = result
        runs[arm] = results
    assert identities["socket"] == identities["shca"]
    comparison = []
    for size, socket in runs["socket"].items():
        shca = runs["shca"][size]
        comparison.append(dict(bytes=size, socket=socket, shca=shca,
            median_shca_over_socket=shca["median_us"] / socket["median_us"],
            p95_shca_over_socket=shca["p95_us"] / socket["p95_us"]))
    glm, pins[GLM] = read(root, GLM)
    ranks = glm["device"]["ranks"]
    assert glm["device_collective_qualified_for_scope"] and len(ranks) == 16
    assert [r["rank"] for r in sorted(ranks, key=lambda rank: rank["rank"])] == list(range(16))
    assert all(r["comm_destroyed"] and r["passed"] and r["step"] == "1103606.66" for r in ranks)
    assert all([(x["operation"], x["elements"]) for x in rank["checks"]]
               == [("allreduce", 1024), ("allgather", 1024), ("allreduce", 6144), ("allgather", 6144)]
               for rank in ranks)
    return dict(source_SHA256=pins, original_steps={k: v[2] for k, v in SOURCES.items()},
        nodes=["f12r2n10", "f12r2n11"], same_nodes_PCI_KFD_boot_Torch_HIP=True,
        comparison=comparison, same_session_pair=False, payload_unit="BF16 input bytes per rank",
        timing_boundary="Max of both ranks for each sample. Host call through device completion, input copy excluded.",
        candidate_consumer=dict(GLM_original_step="1103606.66", nodes=sorted({r["node"] for r in ranks}),
             ranks=16, native_operations=["allreduce", "allgather"], input_bytes_per_rank=[2048, 12288],
             native_provider_SHA256=sorted({r["library_SHA256"] for r in ranks}),
             exact_workloads_matched=0, original_transport=glm["transport_classification"]),
        next_scope="A same-allocation TP16 InfiniCCL comparison at 2048 and 12288 bytes would test the "
                   "actual caller's operations and transport. The original I02 nodes remain reserved while .81 runs.",
        GDR_qualified=False, native_transport_promoted=False, end_to_end_gain_measured=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    print(json.dumps(audit(parser.parse_args().repo), indent=2, allow_nan=False))
