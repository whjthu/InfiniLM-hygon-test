"""Only UNKNOWN/missing and context rejection; no fabricated full800 model PASS fixture."""
import importlib.util
import json
from pathlib import Path

D = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('control_saved_reader', D/'read_original.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    context = dict(request_id='a'*32, publication='b'*40, source_manifest_SHA256='c'*64, epoch=100.)
    bundle = dict(context=context, remote_writes=0, Torch_GPU_calls=0, state_body_reads=0,
                  original_execution_state='UNKNOWN', files={})
    missing = module.audit(bundle, context)
    assert missing['engineering_metadata_verified'] is False and missing['observed_microbatches'] == 0
    assert missing['planned_global_microbatches'] == 800 and missing['missing_files'] == ['run/outcome.json']
    rejected = []
    for label, updates in [('wrong-original-context', {'context': dict(context, request_id='0'*32)}),
                           ('state-body-read-not-this-scope', {'state_body_reads': 1})]:
        try:
            module.audit(dict(bundle, **updates), context)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    print(json.dumps(dict(passed=True, valid_UNKNOWN_missing=1, rejected=rejected,
                          fabricated_complete_model_fixture=False, Torch_GPU_calls=0, remote_calls=0)))


if __name__ == '__main__':
    main()
