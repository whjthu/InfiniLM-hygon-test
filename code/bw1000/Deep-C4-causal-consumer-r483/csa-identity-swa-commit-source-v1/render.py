"""Render the one-TU Deep CSA identity-commit candidate from archived inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NATIVE = "code/bw1000/Deep-PD-disaggregation-r382/native/original/"
CSA = "code/bw1000/cache-pp4-and-deep-profile/Deep-attention-inner-private-r9193b/deepseek_v4_csa_attention_batch_r784.cpp"
STORE = NATIVE + "deepseek_v4_paged_latent_cache_public.cpp"
PREPARE = NATIVE + "deepseek_v4_attention_prepare.cpp"
PLAN = "code/bw1000/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/BUILD_PLAN.json"
PRODUCER = "analysis/bw1000/Deep482-CPU1113-original19-build-and-root-MD-object-ELF-closure.json"
PROFILE = "analysis/bw1000/Deep-r467-post1498-main-phase-decision.json"
PINS = {
    CSA: "2611335a7d57ee12411f3d7d5f076bf138f6db17d8607fec184e8d209bff8c8c",
    STORE: "e7ca7d495c6bed852158cf740a2ab3e846d2922e6de0043363ae7c860adf24d2",
    PREPARE: "3f388df12b9f8884d72cbe6d4a29f2b273934a3dd0cc49c81c93b66fe67ba56b",
    PLAN: "d43a20a26b6b7d0ac72e92882c5771112a0a47ba91e5fff8c15de2b2d8903117",
    PRODUCER: "3a7f17eea35dda4338b0f197d8602dfd7ebee431c0553cf762b2cd93b158946e",
    PROFILE: "7ffef1122f09ecaaf319da3f55c9e7b76d21ceaf4e086a99106a9016691c2da4",
}
HEADER = "identity_swa_commit.hpp"
FLAG = "INFINILM_DEEP_CSA_IDENTITY_SWA_COMMIT"
OLD = ("auto commit=work.alloc({commit_rows.size(),512},rt::DataType::kBFloat16);"
       "ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),"
       "prep.kv,device_ids(commit_rows),int64_t{0},commit);cache_->store(r,device_ids(commit_slots),commit);")
NEW = f"""static const bool identity_commit_opt_in = deep_identity_swa_commit::enabled(std::getenv("{FLAG}"));
 rt::Tensor commit;
 if (deep_identity_swa_commit::can_use_prepared_kv(
         identity_commit_opt_in,
         identity_commit_opt_in && prep.kv && prep.kv->device() == r.device()
             && prep.kv->dtype() == rt::DataType::kBFloat16
             && prep.kv->shape() == rt::Shape{{tokens,512}} && prep.kv->is_contiguous(),
         commit_rows, commit_slots.size(), tokens)) {{
    commit = prep.kv;
 }} else {{
    auto copy = work.alloc({{commit_rows.size(),512}},rt::DataType::kBFloat16);
    ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),
                                       prep.kv,device_ids(commit_rows),int64_t{{0}},copy);
    commit = copy;
 }}
 cache_->store(r,device_ids(commit_slots),commit);"""


def sha(raw):
    return sha256(raw).hexdigest()


def once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError("frozen CSA transform anchor differs")
    return text.replace(anchor, replacement)


def source_bytes():
    original = {path: (REPO / path).read_bytes() for path in PINS}
    if {name: sha(raw) for name, raw in original.items()} != PINS:
        raise ValueError("archived CSA or producer evidence differs")
    csa = original[CSA].decode()
    if ("staging.store(r,staged_ids,prep.kv)" not in csa
            or "work.held.insert(work.held.end(),{prep.query,prep.kv,prep.q_lora})" not in csa
            or b"r.syncStream();return {query,key,ql};" not in original[PREPARE]
            or b"invoke(r,data_,slots,values,flag_,block_,0)" not in original[STORE]):
        raise ValueError("original prepared tensor, store or lifetime binding differs")
    unit, = [item for item in json.loads(original[PLAN])["units"] if item["name"] == "TU07"]
    build = json.loads(original[PRODUCER])
    if not (unit["kind"] == "HOST" and unit["source_SHA256"] == unit["original_source_SHA256"] == PINS[CSA]
            and build["actual_CPU"] == "910407.1113" and build["root_independent"]["passed"] is True
            and build["root_independent"]["original19_M_MD_sets"] is True
            and unit["name"] in build["producer"]["class_consumers"]):
        raise ValueError("CSA source did not bind the original HOST TU07 qualification")
    added = f'#include "{HEADER}"\n#include <cstdlib>\n'
    patched = once(csa, OLD, NEW)
    patched = once(patched, '#include <algorithm>\n', added + '#include <algorithm>\n')
    profile = json.loads(original[PROFILE])
    return patched.encode(), dict(
        pinned_inputs=PINS, cpu1113_parent_library=build["library"], original_TU=unit["name"],
        original_TU_flags_SHA256=sha(json.dumps(unit["prefix"], separators=(",", ":")).encode()),
        measured_commit_cache_enclosures_ms=[dict(decode_call=w["decode_call"], past=w["past"],
            stream_ms=w["attention_inner_siblings_stream_ms"]["commit_cache"]) for w in profile["windows"]],
        generated_candidate_default_off=True, opt_in=FLAG, extra_source_dependency=HEADER,
        planned_target_changed_host_translation_units=1, target_fresh_M_MD_verified=False,
        target_compiles=0, actual_GPU_steps=0, GPU_GO=False, Model_GO=False,
        source_cost_for_one_committed_token=dict(IndexSelect_submissions=1, BF16_temp_bytes=1024,
            I64_index_upload_bytes=8, logical_IndexSelect_read_bytes=1024, logical_IndexSelect_write_bytes=1024))


def render(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source, proof = source_bytes()
    files = {Path(CSA).name: source, HEADER: (HERE / HEADER).read_bytes()}
    proof["generated_source_SHA256"] = {name: sha(raw) for name, raw in files.items()}
    files["SOURCE_PROOF.json"] = (json.dumps(proof, sort_keys=True, indent=2) + "\n").encode()
    pins = {name: sha(raw) for name, raw in files.items()}
    index = (json.dumps(dict(files_SHA256=pins), sort_keys=True, indent=2) + "\n").encode()
    with TemporaryDirectory(dir=output.parent, prefix="deep-identity-commit-") as folder:
        prepared = Path(folder) / "rendered"
        prepared.mkdir()
        for name, raw in {**files, "SOURCE_FILES.json": index}.items():
            (prepared / name).write_bytes(raw)
        prepared.rename(output)
    return dict(source_SHA256=pins[Path(CSA).name], target_source_manifest_SHA256=sha(index), files=len(pins))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_source_dir", type=Path)
    print(json.dumps(render(parser.parse_args().new_source_dir), sort_keys=True, indent=2))
