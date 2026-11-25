"""Verify the original .703 recorder versus the exact full native model path."""
from hashlib import sha256
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE="code/bw1000/"
TRACE=BASE+"Deep-C4-causal-consumer-r483/csa-identity-swa-callsite-trace-v1/"
ORIG=BASE+"Deep-PD-disaggregation-r382/native/original/"
ARCHIVE="results/bw1000/2026-11-24/"
PINS={
    "terminal":(ARCHIVE+"Deep-CSA-identity-SWA-callsite-trace-original703-CPU-terminal.json","bb241cbb2bc82bf2e279315b7c0da441052f4a4de2ecd2887a7db90c7abb5b96"),
    "original_build":(BASE+"Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/BUILD_PLAN.json","d43a20a26b6b7d0ac72e92882c5771112a0a47ba91e5fff8c15de2b2d8903117"),
    "recorder":(TRACE+"commit_trace.hpp","bbbeb9f301385acfe9cb3fb91475782234647fd8c307b43b1233031ec03bfa2b"),
    "original_cpp_negative_test":(TRACE+"check_trace.cpp","f9d142eb4431e5b46add8aa34a440cb79902aafb0951bf4d93ce5ddaa9d179e9"),
    "native_model":(ORIG+"native_sparse_model_r22.cpp","2d68978d64c15daac6decef2d0547eee753441ff2e99d512aea08b19926572d2"),
    "full_decoder_stack":(ORIG+"deepseek_v4_routed_text_model_r20.cpp","7bb82c58e7e7a72f0f06d460d5510d7fc2cf1cb36e1de877b4f590b6b1d3cc8a"),
    "decoder_dispatch":(ORIG+"deepseek_v4_native_routed_decoder_r20.cpp","1886f34a86308859e521778326e3e1119d93ff8e4d85e1ff1f2a3af3b3d6dfa9"),
    "existing_engine_CABI":(ORIG+"modern_native_engine_probe_r22.cpp","8c82b8b6d4d4b4a37d795a7d45983331775cea2b8685961c7d16cc600cd5ff23"),
    "original_model_config_projection":(BASE+"Deep-a314-C1-C4-cost-budget-r463/PINNED_METADATA.json","359320215a59a1f1901acca74bb66ae1ed9d1a709074621b17d8213366073f1e"),
    "component_weight_lineage":(ARCHIVE+"Deep-CSA-layer2-range-binding-16-of-23-original-reconciled.json","871e5056f8e19430fa9ea470b888fda13e8805b424a8d71884b67a3ed65ba3f7"),
}


def read(name):
    path,wanted=PINS[name]
    raw=(ROOT/path).read_bytes()
    if sha256(raw).hexdigest()!=wanted:
        raise ValueError("original source changed: "+name)
    return raw.decode()


def exact(text,place):
    hits=[i for i,line in enumerate(text.splitlines(keepends=True),1) if place in line]
    if len(hits)!=1:
        raise ValueError("missing or ambiguous native tape")
    return hits[0]


def build():
    text={name:read(name) for name in PINS}
    terminal=json.loads(text["terminal"])
    observed=terminal["verified_original"]
    assert observed["accounting"][0:1]==["1194900.703"] and observed["accounting"][2:4]==["COMPLETED","0:0"]
    assert observed["actual_GPU_steps"]==terminal["actual_model_or_weight_calls"]==0 and terminal["trace_only"]
    plan=json.loads(text["original_build"])
    items={x["name"]:x for x in plan["units"]}
    for tu,role in (("TU03","native_model"),("TU11","full_decoder_stack"),("TU38","decoder_dispatch")):
        assert items[tu]["source_SHA256"]==PINS[role][1]
    config=json.loads(text["original_model_config_projection"])
    n=config["model"]["num_hidden_layers"]
    m=config["model"]["compress_ratios"][:n]
    layers=[i for i,ratio in enumerate(m) if ratio==4]
    assert n==43 and layers==list(range(2,43,2)) and (config["TP"],config["PP"])==(8,1)
    component=json.loads(text["component_weight_lineage"])
    assert component["typed_layer"]["TP"]==1 and component["bound_actual_range_value_digests"]==16
    assert component["unbound_actual_range_value_digests"]==7 and component["original_seven"]["execution_state"]=="UNKNOWN"
    assert not component["native_CsaAttention_runtime_adapter_attested_for_this_loader"]
    points={
        "rank_runtime":("native_model","auto&r=fc.require_runtime();check_owner(r);\n"),
        "rank_topology":("native_model","Native sparse requires Hygon TP8"),
        "real_model_forward":("native_model","auto out=deep_->forward(r"),
        "fixed_layer_constructor":("full_decoder_stack","for(size_t layer=0;layer<43"),
        "same_runtime_each_layer":("full_decoder_stack","auto detail=layer->forward(r"),
        "ratio_dispatch":("decoder_dispatch","else if(ratio_==4){csa_="),
        "same_runtime_real_CSA":("decoder_dispatch","else if(csa_)out.attention_value=csa_->forward(r"),
        "engine_context":("existing_engine_CABI","struct Context{"),
        "engine_forward":("existing_engine_CABI","auto output=c->engine->forward(input)"),
        "same_owner_second_invalid":("recorder","state.phase = Phase::invalid;"),
        "reader_rejects_nonfinished":("recorder","state.phase != Phase::finished"),
        "original_duplicate_CXX_negative_fixture":("original_cpp_negative_test","trace::Forward duplicate(&owner"),
    }
    return {
        "decision":"EXISTING_703_ABI_CANNOT_ATTEST_FULL_NATIVE_MODEL_FORWARD",
        "source_pins":{k:dict(path=v[0],SHA256=v[1]) for k,v in PINS.items()},
        "source_locations":{k:dict(source=v[0],line=exact(text[v[0]],v[1])) for k,v in points.items()},
        "original_library_SHA256":observed["library_SHA256"],
        "static_source_expectation_only":{"TP":8,"PP":1,"full_model_layers":n,"same_owner_csa_per_rank":len(layers),
                                           "all_rank_calls_if_one_full_forward":len(layers)*8},
        "recording":{"existing_rows_per_armed_owner":1,"second_same_owner_invalidates":True,
                     "traces_actual":0,"store_returns_actual":0,"stream_drains_actual":0,"owner_threads_actual":0,
                     "capacity_or_drops_measured_actual":False},
        "TP1_component":{"genuine_range_values":16,"required_range_values":23,"known_value_bytes":157460736,
                         "missing_value_bytes":13144320,"is_full_model":False,"runtime_load_and_forward_attested":False},
        "remaining":["Choose a new separately reviewed bounded per-rank trace with explicit overflow/drop and exact same-thread Runtime/worker binding if full Model is required.",
                     "Confirm fresh topology, original input/caller, original provider and unique dispatch independent of historical compilation.",
                     "A TP1 component attempt first requires reconciling the seven original range identities and supplying the actual loaded native adapter."],
        "actual_model_forward":0,"actual_native_CSA_calls":0,"new_GPU":0,"performance_qualification":False,
    }


if __name__=="__main__":
    print(json.dumps(build(),sort_keys=True,separators=(",",":")))
