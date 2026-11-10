"""Check the public projection with synthetic terminal audit inputs."""
import json
from pathlib import Path

from summarize_original import GPU, KEEPER, READER_SHA, digest, summarize

binding = (Path(__file__).resolve().parents[1] / "full-model-screen32-v1" / "INPUT_BINDING.json").read_bytes()
plan = json.loads(binding)["private_plan"]

def accounting(job, state, code):
    return dict(JobIDRaw=job, State=state, normalized_state=state, ExitCode=code, ElapsedRaw=2)

def row(index, rate, **flags):
    item = plan["requests"][index - 1]
    return dict(index=index, diagnostic=False, screen_index=index, ordinal=item["ordinal"],
                suite=item["suite"], backend_decode_tok_s=rate,
                **(dict(correct=False, parse_failure=False, natural_EOS=True, cap_hit=False) | flags),
                private_output="fixture-must-stay-private")

audit = dict(schema="GLM906-original-screen32-root-v1", reader_SHA256=READER_SHA,
             GPU=GPU, keeper=KEEPER, private_plan_SHA256=plan["SHA256"],
             planned_requests=33, unique_screen_planned=32, diagnostic_planned=1,
             GPU_accounting=accounting(GPU, "COMPLETED", "0:0"),
             keeper_accounting=accounting(KEEPER, "FAILED", "1:0"),
             GPU_accounting_success=True, keeper_accounting_success=False,
             keeper_original_producer_passed=False, engineering_full_entry_passed=False,
             all16_original_owner_validation_passed=False, post_clean=True,
             root_requests=[dict(index=0, diagnostic=True, screen_index=None, rate=999),
                            row(1, 15, correct=True), row(2, 15.01),
                            row(17, None, parse_failure=True, natural_EOS=False, cap_hit=True)],
             screen32=dict(planned=32, validated=3, correct=1, parse_failure=1,
                           natural_EOS=2, cap_hit=1, missing_or_unvalidated=29, strict_gt15=1),
             cross_validation_errors=["fixture-must-stay-private"])

def run(value):
    data = json.dumps(value).encode()
    return summarize(data, digest(data), binding)

public = run(audit)
assert public["screen32"]["planned"] == 32 and public["screen32"]["missing_or_unvalidated"] == 29
assert public["suites"]["gsm8k"]["backend_strict_gt15"] == 1
assert public["suites"]["mmlu"]["backend_decode_rate_available"] == 0
assert public["audit_gates"]["post_clean"] and not public["audit_gates"]["keeper_accounting_success"]
assert "fixture-must-stay-private" not in json.dumps(public)

bad = []
for name, value in (("reader_SHA256", "0" * 64), ("engineering_full_entry_passed", True),
                    ("keeper_accounting_success", True)):
    bad.append(audit | {name: value})
bad.append(audit | {"root_requests": audit["root_requests"] + [row(1, 4)]})
bad.append(audit | {"root_requests": [row(1, float("nan"))]})
bad.append(audit | {"root_requests": [row(1, 5) | {"suite": "mmlu"}]})
bad.append(audit | {"screen32": audit["screen32"] | {"strict_gt15": 2}})
bad.append(audit | {"keeper_accounting": accounting(KEEPER, "RUNNING", "0:0")})
for invalid in bad:
    try:
        run(invalid)
    except ValueError:
        continue
    raise AssertionError("invalid GLM906 audit was published")

print(json.dumps(dict(partial_fixture=True, planned=32, validated=3, backend_strict_gt15=1,
                      rejected=len(bad), private_marker_excluded=True, actual_906_consumed=False)))
