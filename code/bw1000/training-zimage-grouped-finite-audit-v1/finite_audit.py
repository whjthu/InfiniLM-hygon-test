"""Batch scalar synchronization for the original per-tensor finite checks."""

import time


class FiniteAudit:
    def __init__(self, torch, step):
        self.torch = torch
        self.order = {6: ("original", "grouped"), 7: ("grouped", "original"),
                      8: ("original", "grouped")}.get(step)
        self.pairs = {}

    def original(self, tensors):
        return all(bool(self.torch.isfinite(tensor).all()) for tensor in tensors)

    def grouped(self, tensors):
        return bool(self.torch.stack([self.torch.isfinite(tensor).all() for tensor in tensors]).all())

    def check(self, values, stage):
        tensors = tuple(values)
        if len(tensors) != 476 or any(tensor is None for tensor in tensors):
            return False
        if self.order is None:
            return self.grouped(tensors)
        if stage in self.pairs:
            raise ValueError("Repeated finite check stage")
        timings, results = {}, {}
        for name in self.order:
            self.torch.cuda.synchronize()
            start = time.perf_counter_ns()
            results[name] = getattr(self, name)(tensors)
            timings[name] = (time.perf_counter_ns() - start) / 1e9
        self.pairs[stage] = {"count": len(tensors), "order": self.order, "seconds": timings,
                             "identical": results["grouped"] == results["original"]}
        if not self.pairs[stage]["identical"]:
            raise ValueError("Grouped finite audit disagrees with the original check")
        return results["grouped"]
