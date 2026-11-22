"""Preserve a complete depth9/15 denominator and apply a component stopping rule."""

from hashlib import sha256
import json
import math
from statistics import median


NODES = ('f12r3n17', 'f12r3n18', 'f12r2n10', 'f12r2n13')
PREVIOUS_POST = dict(zip(NODES, ('1087656.799', '1087656.800', '1103606.817', '1103606.818'), strict=True))
PER_CARD_PAIRS = 'per_card_depth15_over_depth9_pairs'
MIN_POOLED = 1.03
MIN_POSITION = 1.01
MIN_NODE = 1.01


def baseline(raw, wanted):
    report = json.loads(raw)
    actual = report['aggregate_by_mode']['depth9']
    host = actual['normalized_host_ns_per_completed_pair']
    busy = actual['local_driver_busy']['mean_percent']
    pair = report['pair_comparison']
    if not (len(raw) == 15087 and sha256(raw).hexdigest() == wanted
            and report['request'] == '1c5864c1743194e508f5aa2e9cdce87e' and report['qualified'] is True
            and report['planned_cards'] == report['actual_valid_cards'] == report['closed_cards'] == 600
            and report['closed_shards'] == 31 and report['held_shards'] == 0
            and set(NODES).issubset(node for shard in report['original_shards'] for node in shard['nodes'])
            and pair['planned_pairs'] == pair['measured_pairs'] == 1200 and pair['qualified_nodes'] == 75
            and (report['completed_arms'], report['actual_busy_card_samples']) == (3000, 36000)
            and busy == 94.63958333333333 and pair['median_depth9_over_depth3'] == 1.4623816057676526):
        raise ValueError('original closed 600-card depth9 performance evidence differs')
    return dict(local_driver_busy_percent=busy, host_submit_ns_per_pair=host['host_submit_ns'],
                host_query_ns_per_pair=host['host_event_query_ns'], host_sync_wait_ns_per_pair=host['host_sync_wait_ns'],
                median_depth9_over_depth3=pair['median_depth9_over_depth3'])


def representatives(raw, wanted):
    report = json.loads(raw)
    if not (len(raw) == 15277 and sha256(raw).hexdigest() == wanted
            and report['qualified'] is True and report['actual_valid_cards'] == 32
            and report['pair_comparison']['measured_pairs'] == 64
            and report['independent_post_steps'] == PREVIOUS_POST
            and all(row['post_clean'] is True for row in report['close_by_original_label'].values())):
        raise ValueError('prior four representative nodes or their independent cleanup differed')


def classify(campaign):
    decision = dict(passed=False, qualification='INCOMPLETE', median_depth15_over_depth9=None,
                    measured_pairs=0, planned_pairs=64, planned_cards=32,
                    min_pooled_ratio=MIN_POOLED, min_position_ratio=MIN_POSITION,
                    min_each_node_ratio=MIN_NODE, capture_payback_window_min=48,
                    independent_post_required=True, authorize_fleet_expansion=False, formal_model_qualified=False)
    nodes = campaign['nodes']
    pair = campaign['paired_comparison']
    decision['measured_pairs'] = pair['measured_pairs']
    if not (campaign['device_qualified'] is True and set(nodes) == set(NODES)
            and campaign['planned_cards'] == campaign['valid_cards'] == 32
            and campaign['planned_arms'] == campaign['completed_arms'] == 128
            and campaign['planned_busy_card_samples'] == campaign['actual_busy_card_samples'] == 768
            and pair['planned_pairs'] == pair['measured_pairs'] == 64 and pair['qualified_nodes'] == 4):
        return decision
    by_node, positions, fractions = {}, [[], []], []
    for node in NODES:
        row = nodes[node]
        ratios = row[PER_CARD_PAIRS]
        payback = row['capture_window_fraction']
        if row['passed'] is not True or len(ratios) != 8 or set(ratios) != set(payback):
            return decision
        if not all(type(values) is list and len(values) == 2 and all(
                type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values)
                   for values in ratios.values()):
            return decision
        if not all(type(values) is list and len(values) == 2 and all(v is None or (
                type(v) in (float, int) and math.isfinite(v) and v >= 0) for v in values)
                   for values in payback.values()):
            return decision
        by_node[node] = median(value for per_card in ratios.values() for value in per_card)
        for value in ratios.values():
            for position, ratio in enumerate(value):
                positions[position].append(ratio)
        fractions.extend(value for per_card in payback.values() for value in per_card if value is not None)
    pooled = median(positions[0]+positions[1])
    by_position = dict(a_then_b=median(positions[0]), b_then_a=median(positions[1]))
    # A pair with no speedup can have no finite capture payback. Preserve it
    # in the denominator, and require enough measured evidence for the gate.
    paid = sum(fraction <= 1 for fraction in fractions)
    speed = pooled >= MIN_POOLED and min(by_node.values()) >= MIN_NODE and min(by_position.values()) >= MIN_POSITION
    qualified = speed and paid >= 48
    decision.update(passed=qualified, qualification=('KEEP_COMPONENT_CANDIDATE' if qualified
                                                   else 'STOP_AT_DEPTH9'),
                    median_depth15_over_depth9=pooled, median_by_node=by_node, median_by_pair_order=by_position,
                    finite_capture_payback_pairs=len(fractions), capture_payback_within_window_pairs=paid,
                    median_finite_capture_fraction=median(fractions) if fractions else None)
    return decision
