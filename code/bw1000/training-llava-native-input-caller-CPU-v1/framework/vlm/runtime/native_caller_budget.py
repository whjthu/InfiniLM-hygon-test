"""Conservative full-artifact planning from actual native collator observations."""
from __future__ import annotations


def describe_capture(rows, *, parameters, tensors, hidden_size, vocab_size):
    """Report the full payload bound even when the unchanged recorder cannot fit it."""
    if (len(rows) != 64 or [r['ordinal'] for r in rows] != list(range(1, 65))
            or any(type(x) is not int or x <= 0 for x in (parameters, tensors, hidden_size, vocab_size))):
        raise ValueError('complete native caller and parameter denominator required')
    inputs, outputs = [], []
    for row in rows:
        values = [row[k] for k in ('sequence_length', 'selected_logit_rows',
                  'collated_serialized_bytes', 'image_tokens')]
        if (any(type(x) is not int or x <= 0 for x in values)
                or not 0 < values[0] <= 4096 or values[1] > values[0]+1):
            raise ValueError('invalid native collator observation')
        length, selected, serialized, image_tokens = values
        # Include both original kwargs and a full F32 embedding, regardless of
        # whether the real post-encode hook replaces the former with the latter.
        inputs.append(serialized + length*hidden_size*4)
        # F32 logits plus F32 image features even if the latter are absent in the
        # actual result. Hidden-state stacks/attention maps/cache must be OFF.
        outputs.append(selected*vocab_size*4 + image_tokens*hidden_size*4 + 4)
    parameter_bytes = parameters*4
    reserve = 8*1024**2  # every one of25 files: zip metadata,RNG,optimizer scalars
    maximum_artifact = max(parameter_bytes*3, max(inputs), max(outputs))+reserve
    total = parameter_bytes*15 + 8*max(inputs) + 8*max(outputs) + 25*reserve
    fits = maximum_artifact <= 2*1024**3-1 and total <= 8*1024**3
    return dict(rows=64,parameter_tensors=tensors,parameter_elements=parameters,parameter_bytes=parameter_bytes,
                full_parameter_and_AdamW_payload_bytes=15*parameter_bytes,
                max_input_payload_upper_bytes=max(inputs),max_output_payload_upper_bytes=max(outputs),
                metadata_RNG_scalar_reserve_per_file=reserve,planned_files_per_rank=25,
                planned_micro_forwards_per_rank=8,planned_optimizer_updates=2,
                maximum_artifact_upper_bytes=maximum_artifact,per_artifact_limit=2*1024**3-1,
                per_rank_total_upper_bytes=total,per_rank_total_limit=8*1024**3,
                within_recorder_limits=fits,analytical_sufficiency=fits,
                per_artifact_excess_upper_bytes=max(0,maximum_artifact-(2*1024**3-1)),
                per_rank_excess_upper_bytes=max(0,total-8*1024**3),
                actual_GPU_serialized_artifact_bytes_qualified=False,
                input_truncation_allowed=False,output_truncation_allowed=False)


def plan_capture(rows, *, parameters, tensors, hidden_size, vocab_size):
    plan = describe_capture(rows, parameters=parameters, tensors=tensors,
                            hidden_size=hidden_size, vocab_size=vocab_size)
    if not plan['within_recorder_limits']:
        raise ValueError('complete first-two-update capture exceeds unchanged recorder limits')
    return plan


def validate_capture_conditions(model, conditions):
    if conditions != dict(case='T03.sft',hidden_size=4096,vocab_size=151936,
                          language_cache=False,hidden_state_stacks=False,attention_maps=False):
        raise ValueError('unqualified native capture conditions')
    core = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if type(core).__name__ != 'InternVLForConditionalGeneration':
        raise ValueError('native capture model differs')
    language = core.model.language_model
    if (core.config.text_config.hidden_size != 4096 or core.config.text_config.vocab_size != 151936
            or language.config.use_cache is not False):
        raise ValueError('native capture width,vocabulary or language cache differs')
    for config in (core.config,language.config):
        if config.output_hidden_states is not False or config.output_attentions is not False:
            raise ValueError('unbounded hidden states or attention maps requested')
    return dict(passed=True,**conditions,actual_model=type(core).__name__)
