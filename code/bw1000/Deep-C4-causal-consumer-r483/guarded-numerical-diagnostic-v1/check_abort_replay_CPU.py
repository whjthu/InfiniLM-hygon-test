"""Run the guarded diagnostic through its abort and same-input T1 replay."""
from copy import deepcopy
import json
import struct
from unittest.mock import patch

import checkpoint
from check_diagnostic_CPU import GuardRecording, binding, metadata, pins, run_diagnostic
from exercise import summarize_abort_replay


class ReplayRecording(GuardRecording):
    def __init__(self, difference):
        super().__init__()
        self.difference = difference

    def tensor(self, name):
        desc, data = super().tensor(name)
        if self.past == 8496 and self.pending == 1 and self.counts["aborts"] == 1:
            if self.difference == "scores" and name == "scores":
                values = list(struct.unpack("<2124f", data))
                # Change a low unselected score and preserve descending TopK.
                values[0] = 0.5
                data = struct.pack("<2124f", *values)
            elif self.difference == "tie" and name == "indices":
                ids = list(struct.unpack("<512q", data))
                assert ids[0] % 4 == ids[1] % 4
                ids[0], ids[1] = ids[1], ids[0]
                data = struct.pack("<512q", *ids)
        return desc, data


outcomes = {}
for difference, expected in ((None, "identical_output_bytes"),
                             ("tie", "same_scores_topk_ties"),
                             ("scores", "score_bytes_differ")):
    events = []
    with patch.object(checkpoint, "headers", return_value=None), patch.object(
        checkpoint, "read_weight", side_effect=lambda _, item: bytes(item["bytes"])
    ):
        diagnostic = run_diagnostic(ReplayRecording(difference), metadata, pins, None, binding,
                                    lambda _: None, lambda *args: None, events.append)["diagnostic"]
    result = diagnostic["abort_replay"]
    assert result["outcome"] == expected and result["steps"] == [18, 19]
    assert events[-2] == dict(event="abort_replay", report=result)
    assert events[-1]["event"] == "complete"
    assert sum(event["event"] == "step_complete" for event in events) == 71
    outcomes[expected] = result

bad = deepcopy(diagnostic["records"])
bad[19]["past_before"] += 1
try:
    summarize_abort_replay(bad)
except ValueError as error:
    assert "inputs or schedule" in str(error)
else:
    raise AssertionError("mismatched replay input accepted")

print(json.dumps(dict(cases=outcomes, mismatched_geometry_rejected=True,
                      added_device_copy_bytes=0, actual_device=False)))
