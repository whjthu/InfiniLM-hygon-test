"""Separate DSpark worker slice; no checkpoint download, target load or GPU setup.

Used after the draft backbone and shared target head have produced seven base
logit rows. The original six-tap Modal worker is deliberately not modified.
Target-side five-tap capture and recurrent-state verification remain unqualified.
"""
import math

TARGET_LAYERS = (7, 23, 51, 67, 83)
CONSUMER_LAYERS = tuple(i + 1 for i in TARGET_LAYERS)
HIDDEN_SIZE = 7168
PROPOSALS = 7
VERIFY_WIDTH = 8
STREAM = 'after-layer-next-consumer-attnres-prenorm'
SOURCE_URLS = (
    'https://huggingface.co/RadixArk/Kimi-K3-DSpark/blob/main/config.json',
    'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/dspark.py',
    'https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/dspark_components/dspark_draft.py',
    'https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/python/sglang/srt/models/kimi_k3.py',
)


def check_config(config):
    """Reject the old Modal checkpoint rather than reinterpret its tensors."""
    assert tuple(config['dflash_config']['target_layer_ids']) == TARGET_LAYERS
    assert config['hidden_size'] == HIDDEN_SIZE
    assert config['num_hidden_layers'] == 5
    assert config['block_size'] == PROPOSALS
    assert config['markov_head_type'] == 'vanilla'
    assert config['markov_rank'] == 256
    rope = config['rope_parameters']
    assert rope['rope_theta'] == 10000 and rope['rope_type'] == 'yarn'
    assert rope['factor'] == 16
    return dict(feature_width=len(TARGET_LAYERS)*HIDDEN_SIZE,
                proposal_tokens=PROPOSALS, verify_width=VERIFY_WIDTH,
                mask_token_id=config['dflash_config']['mask_token_id'],
                target_layers=list(TARGET_LAYERS), consumer_layers=list(CONSUMER_LAYERS))


def check_feature_packet(packet, body):
    """Validate the actual BF16 wire layout before the existing worker unpacks it.

    New target producers must declare the stream convention and ordered taps;
    old six-tap records cannot satisfy this contract by changing only body size.
    """
    assert packet['draft_kind'] == 'kimi-k3-dspark'
    assert tuple(packet['target_layers']) == TARGET_LAYERS
    assert packet['stream_semantics'] == STREAM
    assert packet['W'] == VERIFY_WIDTH
    count = packet['feature_tokens']
    assert isinstance(count, int) and 0 <= count <= 512
    assert len(body) == len(TARGET_LAYERS)*count*HIDDEN_SIZE*2
    return count, len(TARGET_LAYERS)*HIDDEN_SIZE


def markov_greedy_block(base_logits, anchor_tokens, markov_head, pick=None):
    """Backbone once, seven sequential Markov-corrected greedy proposals.

    Compatible with the HF VanillaMarkov.compute_step_bias interface. `pick`
    defaults to the Torch tensor operation; CPU fixtures supply NumPy argmax.
    Return both proposal tensors and the previous-token tensors read by the
    confidence head. No target/gold tokens enter proposal generation.
    """
    assert len(base_logits.shape) == 3 and base_logits.shape[1] == PROPOSALS
    assert anchor_tokens.shape[0] == base_logits.shape[0]
    if pick is None:
        pick = lambda scores: scores.argmax(dim=-1)
    previous = anchor_tokens
    tokens, previous_tokens = [], []
    for step in range(PROPOSALS):
        previous_tokens.append(previous)
        scores = base_logits[:, step, :] + markov_head.compute_step_bias(previous)
        previous = pick(scores)
        tokens.append(previous)
    return tokens, previous_tokens


def confidence_features(hidden, previous_tokens, markov_head, stack, concatenate):
    """Official confidence inputs: final draft hidden plus previous-token embed."""
    assert len(previous_tokens) == PROPOSALS and hidden.shape[1] == PROPOSALS
    previous = stack(previous_tokens, 1)
    embeddings = markov_head.get_prev_embeddings(previous)
    return concatenate((hidden, embeddings), -1)


def decode_gain_budget(expected_committed_tokens, draft_seconds, verify_seconds,
                       replay_seconds, other_seconds):
    costs = (draft_seconds, verify_seconds, replay_seconds, other_seconds)
    assert all(math.isfinite(v) and v >= 0 for v in costs)
    assert expected_committed_tokens > 0 and sum(costs) > 0
    return expected_committed_tokens/sum(costs)


def cpu_fixture():
    """Control/shape check only; no trained checkpoint or acceptance evidence."""
    import numpy as np
    cfg=dict(hidden_size=7168,num_hidden_layers=5,block_size=7,markov_rank=256,
             markov_head_type='vanilla',dflash_config=dict(target_layer_ids=list(TARGET_LAYERS),mask_token_id=163824),
             rope_parameters=dict(rope_theta=10000,rope_type='yarn',factor=16))
    descriptor=check_config(cfg)
    packet=dict(draft_kind='kimi-k3-dspark',target_layers=list(TARGET_LAYERS),stream_semantics=STREAM,W=8,feature_tokens=1)
    check_feature_packet(packet,bytes(5*7168*2))
    old=dict(packet,target_layers=[19,37,54,66,78,90]); rejected=False
    try:check_feature_packet(old,bytes(6*7168*2))
    except AssertionError:rejected=True
    assert rejected
    class Head:
        def __init__(self):self.seen=[]
        def compute_step_bias(self, previous):
            self.seen.append(previous.copy());scores=np.zeros((len(previous),4),dtype=np.float32)
            scores[np.arange(len(previous)),(previous+1)%4]=2
            return scores
        def get_prev_embeddings(self, previous):return np.repeat(previous[...,None],2,-1).astype(np.float32)
    head=Head();base=np.zeros((2,7,4),dtype=np.float32);anchor=np.asarray([0,2],dtype=np.int64)
    tokens,prev=markov_greedy_block(base,anchor,head,pick=lambda a:a.argmax(axis=-1))
    actual=np.stack(tokens,axis=1);expected=(anchor[:,None]+np.arange(1,8))%4
    assert np.array_equal(actual,expected) and len(head.seen)==7
    features=confidence_features(np.zeros((2,7,3),dtype=np.float32),prev,head,np.stack,np.concatenate)
    assert features.shape==(2,7,5) and np.array_equal(features[:,:,3],np.stack(prev,axis=1))
    return dict(passed=True,CPU_only=True,GPU=False,trained_acceptance_measured=False,
                descriptor=descriptor,old_six_tap_rejected=rejected,
                sequential_markov_steps=7,confidence_previous_token_alignment=True,
                scope='Fixture control/shape only; no numerical model/quality/speed claim.',
                remaining=['New actual five-tap target producer and PP transport not implemented here',
                           'Actual checkpoint key/weight loading and Torch forward not exercised',
                           'Target KDA verify/replay and end-to-end final tokens unqualified'])


if __name__ == '__main__':
    import json
    print(json.dumps(cpu_fixture(),ensure_ascii=False))
