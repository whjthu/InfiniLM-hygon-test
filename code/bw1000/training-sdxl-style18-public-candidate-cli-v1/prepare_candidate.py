"""New domain-only trainer from cached Apache source; no network or training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE = 'local-only/training-sdxl-frozen-trainer-source-recovery-v1-original/frozen-patch-candidate.py'
SOURCE_PIN = {'bytes':63989,'SHA256':'f3f93804fe2a6b7e93b3a46b997dd9955707a8d57ef3ea16796bba5f5a853792'}
TRAINER = 'examples/text_to_image/train_text_to_image_lora_sdxl.py'
ALIASES = ('metrics_path','initial_adapter_path','checkpoint_steps','external_resume_from_checkpoint')


def pin(raw):
    return {'bytes':len(raw),'SHA256':hashlib.sha256(raw).hexdigest()}


def revised(raw):
    if pin(raw) != SOURCE_PIN:
        raise ValueError('Exact cached public candidate differs')
    if b'Licensed under the Apache License, Version 2.0' not in raw:
        raise ValueError('Original Apache source header is absent')
    edits = []
    for key in ALIASES:
        old = ('"--genmlu_' + key + '",').encode()
        new = old + (' "--gencuda_' + key + '",').encode()
        edits.append((old,new))
    edits.append((b'torch.mlu.max_memory_reserved()',b'torch.cuda.max_memory_reserved()'))
    result = raw
    for old,new in edits:
        if result.count(old) != 1:
            raise ValueError('Unique new-domain source anchor differs')
        result = result.replace(old,new,1)
    reversed_source = result
    for old,new in reversed(edits):
        if reversed_source.count(new) != 1:
            raise ValueError('Unique reverse anchor differs')
        reversed_source = reversed_source.replace(new,old,1)
    if reversed_source != raw:
        raise ValueError('Non-instrumentation source bytes changed')
    ast.parse(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    # Source stays private/ignored. Root must externally bind source commit/MF.
    output=args.output.resolve()
    output.mkdir(mode=0o700)
    data=revised((args.repo/SOURCE).read_bytes())
    path=output/'candidate-source-root'/TRAINER
    path.parent.mkdir(parents=True)
    with path.open('xb') as stream:stream.write(data)
    receipt={'state':'NEW_DOMAIN_CANDIDATE_SOURCE_PREPARED_NOT_NATIVE_QUALIFIED',
             'source':SOURCE_PIN,'candidate':pin(data),'relative_trainer':TRAINER,
             'Apache_header_preserved':True,'exact_reverse_patch_equals_original':True,
             'CLI_aliases_added':list(ALIASES),'memory_observation_API':'torch.cuda.max_memory_reserved',
             'original_A100_identity_claimed':False,'HTTP_GETs':0,'model_calls':0,
             'optimizer_updates':0,'supports_common_backend_external_resume':False,
             'training_READY':False}
    with (output/'source-receipt.json').open('x',encoding='utf8') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(receipt,sort_keys=True))


if __name__=='__main__':main()
