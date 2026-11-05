"""Derive the two a314 C4 host timing translation units from pinned project originals."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

CSA = "2611335a7d57ee12411f3d7d5f076bf138f6db17d8607fec184e8d209bff8c8c"
INDEXER = "7f2ffe0395475a98be8723bf434aa9f4944cb5336656b6d57de0f93fc6d1187e"
HEADER = "c4_host_split.hpp"

EXPORT = r'''
extern "C" const char *deep_c4_host_split_read_v1() noexcept {
    namespace original = deep_attention_inner_r9193;
    namespace split = deep_c4_host_split;
    namespace model = infinilm::models::deepseek_v4;
    static thread_local std::string output;
    try {
        std::lock_guard<std::mutex> lock(model::wide_r3443::policy_mutex);
        if (original::mode.load() || original::running.load() ||
            model::wide_r3443::in_flight || deep_owner_r8900::window.load()) return nullptr;
        auto rows = nlohmann::json::array();
        for (unsigned d = 0; d < split::devices; ++d) {
            for (unsigned l = 0; l < split::layers; ++l) {
                const auto &parent = original::rows[d][l];
                if (parent.kind != 2) continue;
                const auto &record = split::records[d][l];
                if (!parent.done || !record.complete ||
                    record.original_frame_start != parent.wall[0]) return nullptr;
                std::array<std::int64_t, split::marks - 1> intervals{};
                for (unsigned i = 0; i < intervals.size(); ++i)
                    intervals[i] = record.ns[i + 1] - record.ns[i];
                rows.push_back({{"device", d}, {"layer", l}, {"past", record.past},
                                {"boundary", record.past % 4 == 3}, {"host_ns", intervals}});
            }
        }
        if (rows.size() != 168) return nullptr;
        const std::array<const char *, 9> labels{
            "latent_compressor_prepare", "indexer_validation_and_compressor_prepare",
            "indexer_query_features", "boundary_new_key_features", "quantized_key_history_bank",
            "scores_status_reduce_topk_drain", "indexer_return_and_cleanup",
            "latent_history_bank_and_swa_staging", "selection_readback_and_device_metadata"
        };
        output = nlohmann::json{{"schema", 1}, {"labels", labels}, {"rows", rows}}.dump();
        for (unsigned d = 0; d < split::devices; ++d)
            for (unsigned l = 0; l < split::layers; ++l)
                if (original::rows[d][l].kind == 2) split::records[d][l].complete = false;
        return output.c_str();
    } catch (...) {
        return nullptr;
    }
}
'''

CAPTURE = '''
 const auto* original_row=inner_profile.row;
 const bool split_enabled=original_row&&tokens==1&&requests.size()==1&&
     (requests.front().past+1)/4>topk_;
 deep_c4_host_split::Capture split_capture(split_enabled,
     original_row?original_row->device:0,original_row?original_row->layer:0,
     split_enabled?requests.front().past:0,original_row?original_row->wall[0]:0);
'''


def read_pinned(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Unexpected original source {path}")
    return raw.decode()


def insertions(text, edits):
    result = text
    for anchor, before, after in edits:
        if result.count(anchor) != 1:
            raise ValueError(f"Source anchor is missing or ambiguous: {anchor}")
        result = result.replace(anchor, before + anchor + after, 1)
    return result


def mark(index):
    return f"deep_c4_host_split::mark({index});"


def derive(csa_path, index_path):
    csa = read_pinned(csa_path, CSA)
    indexer = read_pinned(index_path, INDEXER)
    include = f'#include "{HEADER}"\n'
    csa = insertions(csa, (
        (" auto pending=compressor_->prepare(r,hidden,positions,compress_requests);", CAPTURE, "\n " + mark(1)),
        (" auto index_pending=indexer_->prepare(r,hidden,prep.q_lora,positions,compress_requests);", "", "\n " + mark(7)),
        (" query_begin=0;compressed_begin=0;", " " + mark(8) + "\n", ""),
        (" inner_profile.mark(2);", " " + mark(9) + "\n", ""),
    ))
    indexer = insertions(indexer, (
        (" auto flag=work.alloc({1},rt::DataType::kInt32);", " " + mark(2) + "\n", ""),
        (" if(!pending.compression.boundaries.empty()){", " " + mark(3) + "\n", ""),
        (" size_t historical_count=0;for(auto request:requests)historical_count+=request.past/4;", " " + mark(4) + "\n", ""),
        ("  std::vector<int32_t>visible(request.tokens);", "  " + mark(5) + "\n", ""),
        (" r.syncStream();{auto&c=smallt_stats_r9252.at(r.device().index());", "", ""),
    ))
    # End the scoring interval only after its existing final drain.
    drain = " r.syncStream();{auto&c=smallt_stats_r9252.at(r.device().index());"
    indexer = indexer.replace(drain, " r.syncStream();" + mark(6) + "{auto&c=smallt_stats_r9252.at(r.device().index());", 1)
    return include + csa + EXPORT, include + indexer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("csa", "indexer", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    csa, indexer = derive(args.csa, args.indexer)
    args.output.mkdir()
    files = {args.csa.name: csa.encode(), args.indexer.name: indexer.encode(),
             HEADER: Path(__file__).with_name(HEADER).read_bytes()}
    for name, data in files.items():
        with (args.output / name).open("xb") as stream:
            stream.write(data)
    print(json.dumps({name: sha256(data).hexdigest() for name, data in files.items()}))


if __name__ == "__main__":
    main()
