"""Bind the exact completed and independently reviewed metadata-only Swift skip result."""
import json
from pathlib import Path


ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/"
          "internvl-gqa56-native-swift-resume-skip-69052f94c9554014821c2cf81e29ebfa/output")
REQUEST="69052f94c9554014821c2cf81e29ebfa"
RESULT="35854f709f161191cbdaba88b13baf308bb07c329f0dff56a72faf95e7bcdeae"
PLAN="04f7385ba2b79dd42dddedf03f2ec8f3f1848ae4ee13727f8eea0a4a01a4e005"
PLAN_NAME="native-swift-resume-skip-plan.private.json"
PUBLICATION="9163774490938b0f774c26c7b7f78356f45d529d"
MANIFEST="3ff3127fb56546f2789261369dfe4f8449e4845ef13253ea8d68b2ac270cf0aa"
PUBLIC_ARCHIVE="a643cef5046b1a80fbe95963e163c64b9ecb78f0538adad5bed30d50598d88a0"
INDEPENDENT="5d1956adc7ebe5d702e890bf7ca8bfacd602a1bf0600eb048dfcfb05bda72ed8"
ORIGINAL_TRANSPORT="a78322311a74ebb39f2175bb248a8e43748c8be7d40702e97bf6759c8ab04858"
ROW=["1103606.4790","internvl-gqa56-swift-skip-69052f94c955","COMPLETED","0:0","28","f13r1n14",
     "cpu=8,mem=24G,node=1","2026-11-27T20:15:44","2026-11-27T20:16:12"]


def binding():
    return dict(qualified=True,scope="native_Swift_zero_GRES_metadata_only",
                actual_step=ROW[0],result_SHA256=RESULT,plan_SHA256=PLAN,source_manifest_SHA256=MANIFEST,
                source_publication=PUBLICATION,public_archive_SHA256=PUBLIC_ARCHIVE,
                independent_receipt_SHA256=INDEPENDENT,production_worker_equivalence=False)


def require(ok,code):
    if not ok:
        raise ValueError(code)


def validate_result(row):
    expect=dict(request=REQUEST,actual_step=ROW[0],actual_native_metadata_shards_verified=8,
                actual_loader_constructions_per_rank=2,actual_swift_skip_patch_calls_per_rank=1,
                actual_native_epoch1_skipped_sampler_positions=32,actual_native_epoch1_skipped_dataset_fetches=0,
                actual_native_epoch1_retained_fetches=32,actual_set_epoch_calls_per_rank=12,
                data_seed=42,verified_all_epochs_slots=768,verified_continuation_slots=704,
                verified_global_updates_covered=22,verified_epochs=12,observed_loader_workers=0,
                production_workers=4,continuation_plan_SHA256=PLAN,continuation_plan_bytes=141110,
                actual_optimizer_updates=0)
    require(all(row.get(key)==value for key,value in expect.items()) and
            row["prior_4738_byte_identical"] is row["epoch1_byte_matches_4726"] is True and
            row["production_worker_behavior_verified"] is row["actual_forward_order_verified"] is False,
            "EXACT_4790_NATIVE_SKIP_CPU_RESULT")


def verify(here,pin):
    require(json.loads(pin(here/"SWIFT_SKIP_CPU.json",cap=16384))==binding(),
            "ORIGINAL_SWIFT_NATIVE_SKIP_CPU_WITNESS_REQUIRED")
    row=json.loads(pin(ROOT/"result.json",expected=RESULT,extent=10063,cap=11000))
    validate_result(row)
    return pin(ROOT/PLAN_NAME,expected=PLAN,extent=141110,cap=150000)
