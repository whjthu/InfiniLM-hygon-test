# T03-only USGS native CPU v2: original model basename hypothesis

Source-only, root_GO/CPU_GO false. The prior477ed0d5 T03 run is a known terminal guard failure with0/55 processed and clean post; v1 is never replayed or modified. PRIOR_V1_FAILURE.json is actually checked by the new installer before processor execution.

The two newly read exact Swift bodies register OpenGVLab/InternVL3_5-8B-HF in the internvl_hf ModelGroup and register InternvlhfTemplate. They do not include the generic group matcher, and v1 did not record the raw template value. The cause is therefore not proven.

The only candidate data-path change is a metadata-only view retaining the original basename: NEW_SCOPE/model-metadata/InternVL3_5-8B-HF. All13 original metadata bytes/SHA remain unchanged. The namespace is exclusively /home/whj/nvidia-baseline/alignment-assets/training-USGS-T03-native-template-CPU-v2. No processor/model/template metadata is assigned or patched.

The actual get_model_processor call and get_template(template_type=None) call are AST-identical to v1. The worker saves empty_is_None, raw registration, processor/config classes before its strict raw internvl_hf guard. It additionally saves and checks final resolved internvl_hf immediately after the original factory returns. Both native wrappers remain intact; no unwrapped encode leaf or manual tokens are called. Failure still retains all45train/10dev slots.

Root reviews, publishes and invokes the same small install_and_run.py protocol with a new request/epoch/payload. It stages only the small source, existing11PNG/45+10 JSONL/binding, and13 pinned metadata members, while reusing the existing396 source. The controller CLI is:

```text
A100_PYTHON -B run_CPU_once.py --case T03.sft --framework EXISTING396 --native-binding ORIGINAL2FF3 --model-view NEW_SCOPE/model-metadata/InternVL3_5-8B-HF --data NEW_SCOPE/data --data-binding-sha ACTUAL_SHA --data-binding-bytes ACTUAL_BYTES --scope NEW_SCOPE/run --request-id ORIGINAL32HEX --publication PUB40HEX --epoch ORIGINAL_EPOCH
```

Resources and bounds remain2CPU excluding0/1,8GiB observed soft group RSS,0GPU/no weights,300-second origin including install; work260/stop270/held282/registry285/alarm290/outer295. Qualified owned_processes/held_child/wrapper bytes, common396 manifest, original recipe and complete55 inputs are reused unchanged. No new remote/Native/GPU run has occurred.

Only new AST/path-delta checks are run locally; original join/cleanup/native25 tests are not repeated. Raw source bodies remain ignored, not in this package.
