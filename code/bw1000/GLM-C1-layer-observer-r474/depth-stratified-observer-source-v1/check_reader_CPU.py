"""Exercise the actual reader against controlled deep-layer marker envelopes."""
import copy

from attention_evidence import attention_windows
from depth_binding import INDEXER_TYPES, PAIRS
from ffn_evidence import MARKS, ROUTED_MARKS, check_frame, routed_windows


DIAGNOSTIC = dict(prompt_tokens=1312, prefill_fragment_plan=[512, 512, 288])


def fixture(ordinal, rank):
    past = DIAGNOSTIC['prompt_tokens'] + ordinal - 1
    pair = PAIRS[ordinal - 1]
    names = ['CAPI.owner.begin']
    for layer, kind in enumerate(INDEXER_TYPES):
        names.append(f'layer.{layer}.{kind}.begin')
        if layer in pair:
            selected_layer(names, layer, kind, rank, past)
        names.append(f'layer.{layer}.end')
    marks = [dict(name=name, owner_host_s=10 + i * .001, gpu_from_first_s=i * .0009)
             for i, name in enumerate(names)]
    return dict(tokens=1, rank=rank % 8, stage=0, row_past_count=[[0, past, 1]],
                sequence=len(DIAGNOSTIC['prefill_fragment_plan']) + ordinal,
                detail_frame_ordinal=ordinal, full_detail_layer=pair[0], shared_detail_layer=pair[1],
                event_count=len(marks), marks=marks)


def selected_layer(names, layer, kind, rank, past):
    base = f'detail.layer.{layer}.'
    prefix = base + 'attnd1.'

    def window(stem, body=None, placement=prefix):
        names.append(placement + stem + '.begin')
        if body is not None:
            body()
        names.append(placement + stem + '.return')

    def projection(m, n, k, row, col=0, site='projection'):
        names.extend((prefix + f'{site}.tuple.M{m}.N{n}.K{k}.row_offset{row}.column_offset{col}'
                      f'.input_BF16.contiguous_checked.canonical_stride_inferred_{k}_1'
                      '.raw_stride_UNKNOWN.kernel_provider_UNKNOWN',
                      prefix + f'{site}.dispatch.cached_r809.kernel_provider_UNKNOWN'))
        if site == 'projection':
            window('projection.original_success_fence')

    def prepare():
        projection(1, 2048, 6144, 0)
        projection(1, 1024, 2048, rank * 1024)
        projection(1, 576, 6144, 0)

    def indexer():
        projection(1, 128, 6144, 0)
        projection(1, 256, 2048, rank * 256)

    window('prepare.forward_latent', prepare)
    if kind == 'full':
        window('attn.indexer.features', indexer, base)
    window('fresh.expand_cached', lambda: projection(1, 1792, 512, rank * 1792))
    names.extend(prefix + name for name in ('selection.complete_prefix', 'v398.enabled',
                                           'history.dispatch.standard'))
    window('history.selected_value', lambda: projection(past + 1, 1792, 512, rank * 1792))
    window('attention.selected')
    window('attention.result_copy')
    window('output.forward', lambda: projection(1, 6144, 1024, 0, rank * 1024, 'output'))
    window('output.original_stage_finish_or_stream_wait')
    names.append(base + 'FFN.begin')
    names.extend(base + 'FFN.routed.' + name for name in ROUTED_MARKS)
    projection(1, 128, 6144, rank * 128)
    projection(1, 128, 6144, rank * 128)
    names.extend(base + name for name in MARKS)
    names.append(base + 'FFN.end')


def rejected(frame, rank, ordinal):
    try:
        check_frame(frame, DIAGNOSTIC, rank, ordinal - 1)
    except ValueError:
        return
    raise AssertionError('malformed depth evidence was accepted')


def rename(frame, before, after):
    marker = next(mark for mark in frame['marks'] if mark['name'] == before)
    marker['name'] = after


def check():
    for ordinal, rank in enumerate((0, 9, 15), 1):
        frame = fixture(ordinal, rank)
        pair = PAIRS[ordinal - 1]
        assert check_frame(frame, DIAGNOSTIC, rank, ordinal - 1) == list(pair)
        assert [len(report['tuples']) for report in attention_windows(frame, rank)] == [10, 8]
        for layer in pair:
            costs = routed_windows(frame, layer)
            assert len(costs) == 10 and all(x['host_s'] > 0 for x in costs.values())

    frame = fixture(1, 0)
    for replacement in (dict(detail_frame_ordinal=2), dict(full_detail_layer=0),
                        dict(shared_detail_layer=20), dict(rank=1), dict(event_count=455)):
        rejected(dict(frame, **replacement), 0, 1)
    alterations = (
        ('layer.27.shared.begin', 'layer.27.full.begin'),
        ('detail.layer.18.FFN.routed.gateup.begin', 'detail.layer.17.FFN.routed.gateup.begin'),
        ('detail.layer.18.FFN.routed.gateup.begin', 'detail.layer.18.FFN.routed.gateup.end'),
        ('detail.layer.18.attnd1.attention.result_copy.begin',
         'detail.layer.17.attnd1.attention.result_copy.begin'),
    )
    for before, after in alterations:
        broken = copy.deepcopy(frame)
        rename(broken, before, after)
        rejected(broken, 0, 1)
    broken = copy.deepcopy(frame)
    marker = next(m for m in broken['marks'] if 'layer.18.attnd1.projection.tuple.M1.N128.K6144' in m['name']
                  and m['name'].endswith('kernel_provider_UNKNOWN'))
    # The first N128 on the full layer is the indexer; the last ones are the
    # sparse shared expert, whose width cannot be copied from dense layer zero.
    matches = [m for m in broken['marks'] if m['name'] == marker['name']]
    assert len(matches) == 3
    matches[-1]['name'] = matches[-1]['name'].replace('.N128.', '.N768.')
    rejected(broken, 0, 1)
    print('3 complete depth frames passed; 10 malformed frames rejected.')


if __name__ == '__main__':
    check()
