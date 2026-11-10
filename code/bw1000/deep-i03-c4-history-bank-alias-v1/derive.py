"""Derive a default-OFF C1 history-bank alias from the pinned production CSA."""
from hashlib import sha256
from pathlib import Path

CSA_SHA = "2611335a7d57ee12411f3d7d5f076bf138f6db17d8607fec184e8d209bff8c8c"
HEADER = "history_bank_alias.hpp"
EXPORTS = ("deep_c4_history_bank_alias_set_v1", "deep_c4_history_bank_alias_stats_v1")
EXPORT = r'''
extern "C" int deep_c4_history_bank_alias_set_v1(int mode) noexcept {
    try { return deep_c4_history_bank_alias::set(mode); } catch (...) { return 3; }
}
extern "C" int deep_c4_history_bank_alias_stats_v1(
        int device, std::uint64_t *out, std::size_t count) noexcept {
    try { return deep_c4_history_bank_alias::read(device, out, count); }
    catch (...) { return 3; }
}
'''

EDITS = (
    ("try{deep_attention_inner_r9193::Frame inner_profile(r,2,tokens,requests.size());",
     "try{deep_c4_history_bank_alias::Use history_policy(r.device().index());"
     "deep_attention_inner_r9193::Frame inner_profile(r,2,tokens,requests.size());"),
    (' auto compressed_bank=work.alloc({std::max(size_t(1),compressed_total),512},rt::DataType::kBFloat16);\n'
     ' r.setDeviceMemory(compressed_bank->data(),0xff,compressed_bank->nbytes());',
     ' const bool history_eligible=deep_c4_history_bank_alias::eligible(\n'
     '     tokens,requests.size(),requests.front().past,compressed_total,!pending.boundaries.empty());\n'
     ' const bool history_alias=history_policy.enabled&&history_eligible;\n'
     ' rt::Tensor compressed_bank;\n'
     ' if(!history_alias){\n'
     '  compressed_bank=work.alloc({std::max(size_t(1),compressed_total),512},rt::DataType::kBFloat16);\n'
     '  r.setDeviceMemory(compressed_bank->data(),0xff,compressed_bank->nbytes());\n'
     ' }'),
    ('work.held.push_back(old_compressed);compressed_bank->narrow({{0,compressed_begin+start,n}})->copy_from(r,old_compressed);',
     'work.held.push_back(old_compressed);'
     'if(history_alias)compressed_bank=old_compressed;'
     'else compressed_bank->narrow({{0,compressed_begin+start,n}})->copy_from(r,old_compressed);'),
    ('inner_profile.finish_after_original_drain();return projected.hidden;',
     'inner_profile.finish_after_original_drain();'
     'history_policy.complete(history_eligible,history_alias,compressed_total*512*2);'
     'return projected.hidden;'),
)

def derive(path):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != CSA_SHA:
        raise ValueError("Original CSA source differs")
    text = raw.decode()
    for old, new in EDITS:
        if text.count(old) != 1 or new in text:
            raise ValueError("CSA source anchor missing or ambiguous")
        text = text.replace(old, new, 1)
    return f'#include "{HEADER}"\n' + text + EXPORT

def restore(text):
    prefix = f'#include "{HEADER}"\n'
    if not text.startswith(prefix) or not text.endswith(EXPORT):
        raise ValueError("Candidate boundaries differ")
    text = text[len(prefix):-len(EXPORT)]
    for old, new in reversed(EDITS):
        if text.count(new) != 1:
            raise ValueError("Candidate edit is missing or ambiguous")
        text = text.replace(new, old, 1)
    if sha256(text.encode()).hexdigest() != CSA_SHA:
        raise ValueError("Candidate differs outside the declared edits")
    return text.encode()
