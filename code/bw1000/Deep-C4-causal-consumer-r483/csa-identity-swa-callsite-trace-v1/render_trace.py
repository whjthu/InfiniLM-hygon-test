"""Place a bounded observer at the actual SWA commit branch in the candidate TU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
PRIOR = HERE.with_name("csa-identity-swa-commit-source-v1") / "render.py"
PRIOR_SHA = "27551623e83e6f68b038174fa649e037b2f7ef1cd8f4e4073f6ed91660796ad8"
HEADER = "commit_trace.hpp"
CABI = r'''
extern "C" int modern_deep_csa_commit_trace_arm_v1(const void* owner, std::uint64_t tag) noexcept {
    return deep_csa_commit_trace::arm(owner, tag);
}
extern "C" int modern_deep_csa_commit_trace_clear_v1(const void* owner) noexcept {
    return deep_csa_commit_trace::clear(owner);
}
extern "C" int modern_deep_csa_commit_trace_read_v1(const void* owner, std::uint64_t* out, std::size_t size) noexcept {
    return deep_csa_commit_trace::read(owner, out, size);
}
'''


def digest(raw):
    return sha256(raw).hexdigest()


def build():
    prior = ModuleType("qualified_candidate_renderer")
    prior.__file__ = str(PRIOR)
    raw = PRIOR.read_bytes()
    if digest(raw) != PRIOR_SHA:
        raise ValueError("qualified candidate renderer differs")
    exec(compile(raw, str(PRIOR), "exec"), vars(prior))
    before, evidence = prior.source_bytes()
    unit = before.decode()
    # Use an object on the actual forward stack so failures retain only the
    # program points reached before the exception.
    for anchor, replacement in (
        ('#include "identity_swa_commit.hpp"', '#include "identity_swa_commit.hpp"\n#include "commit_trace.hpp"'),
        ('if(total!=tokens)throw std::invalid_argument("DeepSeek CSA packed token count differs");',
         'if(total!=tokens)throw std::invalid_argument("DeepSeek CSA packed token count differs");\n'
         ' deep_csa_commit_trace::Forward commit_trace(&r, tokens, requests.size());'),
        ('    commit = prep.kv;', '    commit = prep.kv;\n    commit_trace.direct(commit_rows.size());'),
        ('    commit = copy;', '    commit_trace.index_select(commit_rows.size());\n    commit = copy;'),
        ('cache_->store(r,device_ids(commit_slots),commit);',
         'cache_->store(r,device_ids(commit_slots),commit);commit_trace.store();'),
        ('r.syncStream();inner_profile.finish_after_original_drain();return projected.hidden;',
         'r.syncStream();inner_profile.finish_after_original_drain();commit_trace.complete_after_drain();return projected.hidden;'),
    ):
        unit = prior.once(unit, anchor, replacement)
    source = (unit + CABI).encode()
    files = {Path(prior.CSA).name: source, HEADER: (HERE / HEADER).read_bytes(),
             prior.HEADER: (PRIOR.parent / prior.HEADER).read_bytes()}
    result = dict(old_source_SHA256=digest(before), source_pins=evidence["pinned_inputs"],
                  files_SHA256={name: digest(data) for name, data in files.items()},
                  owner_thread_only=True, trace_only=True, runtime_GPU_steps=0, GPU_GO=False)
    files["SOURCE_PROOF.json"] = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    return files, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    files, result = build()
    args.destination.mkdir()
    for name, data in files.items():
        with (args.destination / name).open("xb") as stream:
            stream.write(data)
    print(json.dumps(result, sort_keys=True))
