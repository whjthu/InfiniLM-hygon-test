"""Read-only source checks; no compile, subprocess, imports of Native, or device calls.

This proves an explicitly defined lexical normalization, not compiler/runtime
equivalence or performance. Prints JSON only; does not write a result file.
"""
from pathlib import Path
import hashlib
import json
import re

D = Path(__file__).resolve().parent
R = D.parents[3]
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda name: json.loads((D / name).read_text(encoding="utf-8"))
TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27|[A-Za-z_]\w*|\d+(?:\.\d*)?|\S', re.S)


def tokens(text):
    return [t for t in TOKEN.findall(text) if not t.startswith(("//", "/*"))]


def normalize(text):
    # Observation additions are delimited. The remaining production tokens
    # must exactly equal the pinned parent, independently of SOURCE_EDITS.
    def remove(match):
        body = match.group(1)
        stripped = " ".join(t for t in tokens(body) if not t.startswith('"'))
        assert not re.search(r'\b(syncStream|retire|success_fence|memcpyD2H|memcpyH2D|fp8_cached_r809|allreduce_|submit)\b', stripped)
        assert "glm_attention_detail_v1::" in body
        return "\n"
    text = re.sub(r'/\* ATTND1_OBSERVATION_BEGIN \*/(.*?)/\* ATTND1_OBSERVATION_END \*/', remove, text, flags=re.S)
    assert "ATTND1_OBSERVATION" not in text
    text = text.replace('#include "attention_detail_v1.hpp"', '')
    # Exact single-call caching rewrite, two known call expressions only.
    for expression in (
        'fp8_cached_r809(r,out,packed,scale,weight_,scale_,offset_,size_t{0})',
        'fp8_cached_r809(r,local,packed,global_scale,weight_,scale_,size_t{0},offset_)',
    ):
        pattern = r'const bool ad_v1_cached = ' + re.escape(expression) + r';\s*if\(!ad_v1_cached\)'
        text = re.sub(pattern, 'if(!' + expression + ')', text)
    assert 'ad_v1_cached' not in text
    return text


def canonical_lines(text):
    # Whitespace-only publication normalization; production tokens are checked separately.
    return "\n".join(line.rstrip(" \t") for line in text.replace("\r\n", "\n").split("\n"))


def main():
    parents, edits, budget = read('PARENTS.json'), read('SOURCE_EDITS.json')['edits'], read('EVENT_BUDGET.json')
    mf = read('SOURCE_MANIFEST.json')
    assert mf['source_only'] is True and mf['GPU_GO'] is False and mf['target_compile'] is False
    for name, pin in mf['files_SHA256'].items():
        assert H(D / name) == pin, ('package pin', name)
    for name, info in parents['files'].items():
        p = R / info['path_from_repo']
        assert H(p) == info['SHA256'], ('parent pin', name)
        if info.get('byte_identical'):
            assert (D / name).read_bytes() == p.read_bytes(), ('unchanged header', name)
    checked = []
    for name in ('NoV_tp16.cpp', 'prepare_tp16.cpp', 'output_tp16.cpp'):
        parent = (R / parents['files'][name]['path_from_repo']).read_text(encoding='utf-8')
        new = (D / name).read_text(encoding='utf-8')
        assert tokens(normalize(new)) == tokens(parent), ('normalized production tokens', name)
        # Splitting a parent line can turn its original interior indentation
        # into trailing whitespace. Keep an exact replay ledger so inverse
        # restores only those known line-end spaces, never normalizes interior
        # whitespace or changes SOURCE_EDITS.old / parent bytes.
        replay = canonical_lines(parent)
        steps = []
        for e in [e for e in edits if e['file'] == name]:
            assert replay.count(e['old']) == 1, ('edit replay', e['id'])
            untrimmed = replay.replace(e['old'], e['new'], 1)
            after = canonical_lines(untrimmed)
            steps.append((e, replay, untrimmed, after))
            replay = after
        assert replay == new, ('whitespace-normalized edit replay', name)
        restored = new
        for e, before, untrimmed, after in reversed(steps):
            assert restored == after == canonical_lines(untrimmed)
            assert untrimmed.count(e['new']) == 1, ('edit inverse', e['id'])
            restored = untrimmed.replace(e['new'], e['old'], 1)
            assert restored == before
        assert restored == canonical_lines(parent), ('LF-and-trailing-whitespace-normalized inverse', name)
        assert new == canonical_lines(new), ('CPP trailing whitespace', name)
        # Mark additions must not modify any existing destructor body.
        destructors = lambda s: re.findall(r'~Work\(\).*?\}\s*rt::Tensor alloc', s, re.S)
        assert destructors(new) == destructors(parent), ('destructor', name)
        old_tokens, new_tokens = tokens(parent), tokens(new)
        for operation in ('syncStream', 'success_fence', 'retire', 'memcpyD2H', 'memcpyH2D', 'allreduce_', 'fp8_cached_r809'):
            assert new_tokens.count(operation) == old_tokens.count(operation), (name, operation)
        checked.append(name)
    # Verify the cap is a sum of actual added helper call sites, not just a
    # changed caller assertion. Existing markers are preserved by normalization.
    event_call = re.compile(r'glm_attention_detail_v1::(?:mark|tuple|dispatch)\s*\(')
    counted = {(e['file'], e['id']): len(event_call.findall(e['new'])) - len(event_call.findall(e['old'])) for e in edits}
    rows = {(x['file'], x['id']): x for x in budget['new_mark_rows']}
    assert set(rows) == {key for key, count in counted.items() if count}
    for key, row in rows.items():
        assert counted[key] == row['marks_per_execution']
        assert row['max_new_marks'] == counted[key] * row['max_executions_per_frame']
    for item in budget['support_files']:
        p = R / item['path_from_repo']
        assert H(p) == item['SHA256'], ('budget supporting source', str(p))
    support = {item['role']: (R / item['path_from_repo']).read_text(encoding='utf-8') for item in budget['support_files']}
    prepare_parent = (R / parents['files']['prepare_tp16.cpp']['path_from_repo']).read_text(encoding='utf-8')
    nov_parent = (R / parents['files']['NoV_tp16.cpp']['path_from_repo']).read_text(encoding='utf-8')
    latent_calls = len(re.findall(r'\b(?:qa_|qb_|ka_)->forward\(', prepare_parent))
    indexer_calls = len(re.findall(r'\b(?:wk_|wq_)->forward\(', support['native_indexer']))
    fresh_calls = nov_parent.count('auto fresh_all=prepare_->expand_cached(')
    topk = int(re.search(r'topk>(\d+)', nov_parent).group(1))
    block = int(re.search(r'max_block_rows=(\d+)', support['history_chunk']).group(1))
    assert 'if(U<=max_block_rows)' in support['history_selected']
    assert 'auto result=prepare.expand_cached(r,l,k->view({U,1,64}))' in support['history_selected']
    assert 'auto ids=work.ids(logical)' in nov_parent and 'count=request.tokens' in nov_parent
    # Formula derived from call sites: latent qa/qb/ka=3; indexer wk/wq=2;
    # fresh kb=1; C1 selected history <=1. Use the full bound for BOTH layers.
    formula = budget['multiplicity_derivation']
    assert formula == dict(selected_layers=2, latent_projections=latent_calls, indexer_projections_upper=indexer_calls,
                          fresh_projections=fresh_calls, history_projections_upper=(topk + block - 1) // block,
                          selected_rows=1, topk_max=topk, history_block_rows=block)
    projection_calls = formula['selected_layers'] * sum(formula[k] for k in ('latent_projections', 'indexer_projections_upper', 'fresh_projections', 'history_projections_upper'))
    assert projection_calls == 14 and formula['topk_max'] <= formula['history_block_rows']
    for row in rows.values():
        expected = projection_calls if row['id'].startswith('projection.') else (4 if row['id'] == 'expand.original_stream_wait' else 2)
        assert row['max_executions_per_frame'] == expected
    added = sum(r['max_new_marks'] for r in rows.values())
    assert added == budget['new_marks_upper_bound'] == 162
    assert budget['old_caller_upper_bound'] == 278
    assert budget['future_caller_upper_bound'] == 278 + added == 440 <= budget['pool_events_unchanged'] == 512
    helper = (D / 'attention_detail_v1.hpp').read_text(encoding='utf-8')
    assert not any(x in helper for x in ('cudaEventCreate(', 'cudaEventSynchronize(', 'cudaStreamSynchronize(', 'syncStream(', 'success_fence(', 'retire(', 'thread_local', 'extern "C"'))
    assert helper.count('glm_sparse_r131::detail(') == 3
    assert '#include <cstdlib>' in helper
    assert 'std::getenv("GLM_ATTENTION_DETAIL_V1")' in helper
    assert "return optin && optin[0] == '1' && optin[1] == '\\0';" in helper
    assert helper.index('if (!selected_c1) return false;') < helper.index('std::getenv(')
    for condition in ('s->active', 's->enabled', 's->detail', 's->current_layer == 0', 's->current_layer == 3', 's->selector.tokens != 1', 's->selector.rows.size() != 1', 'row[0] != 0', 'row[1] == 0', 'row[2] != 1', 'state->second.get() == s', 'c1_next_decode_r474.find(owner)', 'frames->second >= 1 && frames->second <= 3'):
        assert condition in helper, ('missing guard', condition)
    return dict(source_only=True, GPU_GO=False, target_compile=False,
                normalized_production_token_equality=checked, exact_edit_inverse=True, inverse_normalization="LF plus trailing spaces/tabs only",
                immutable_headers=5, original_sync_dispatch_counts_equal=True,
                Work_destructors_unchanged=True, new_marks_upper_bound=added,
                future_caller_cap=440, event_pool=512, compiled=False, device_executed=False,
                limitation='Static source checks only; not compiler/runtime equivalence, timing, or target closure qualification.')


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
