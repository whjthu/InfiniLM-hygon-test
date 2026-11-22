"""Exact Gemma language regex shell, exactly seven unique literal alternatives only.

No regex widening, search, config rewriting, layer/parameter changes or target discovery.
"""
PREFIX=r'^(model\.language_model(?=\.).*\.('
SUFFIX=r'))$'
ALTERNATIVES=frozenset(('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'))
OBSERVED_ADAPTER_CONFIG_SHA256='3453c728728e4eefb7704b62a4e727278f4ebdf9a94f98b67ba146c8be9a16bc'

def signature(value):
    if type(value) is not str or not value.startswith(PREFIX) or not value.endswith(SUFFIX):
        raise ValueError('exact Gemma anchored language target shell required')
    parts=value[len(PREFIX):-len(SUFFIX)].split('|')
    if len(parts)!=7 or len(set(parts))!=7 or frozenset(parts)!=ALTERNATIVES:
        raise ValueError('exact seven unique Gemma projection alternatives required')
    return tuple(sorted(parts))

def validate_recipe(config,structural):
    if (config['r']!=16 or config['lora_alpha']!=32 or config['lora_dropout']!=.05 or config['bias']!='none'
            or signature(config['target_modules'])!=signature(structural['target_modules'])):
        raise ValueError('same Gemma language LoRA recipe required')
    return dict(target_shell_exact=True,unique_projection_alternatives=7,alternative_set_equal=True,
                config_modified=False,parameters_replaced=False)
