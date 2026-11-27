# MiniCPM OFF six-row native CPU ingress

This source derives a single zero-GRES processor witness from the exact native88 worker. It preserves the original full and 4096 encodes, trained ChatML suffix and EOS checks, supervision mask, singleton collator comparison, nine metadata files, exact Swift and Transformers providers, DTK layout, initial adapter file check and independent terminal evidence. The worker runs processor-only and performs zero optimizer updates.

The only private data members are `off-image-01.jpg` through `off-image-04.jpg`, `train.jsonl` and `INPUT_FILES.json`. The input manifest is 2281 bytes with SHA-256 `9a1338a008b91e2524873e12f49517889a23af4a9869232d5fa1c6103a0e60d4`. The six files total 2,492,792 bytes. Original train-only questions and answers are decoded and retained verbatim in UTF-8. The private manifest binds each ordinal to its original line SHA and opaque image name. The original reviewed manifest is checked locally and never included in the target package.

## Local preparation

Run this from the checkout with the bundled Python on the exact local bundle.

```text
python -B code/bw1000/training-minicpm-off-native6-CPU-v1/check_local.py local-only/minicpm-off-packaging-pilot-20261127
python -B code/bw1000/training-minicpm-off-native6-CPU-v1/prepare.py --source local-only/minicpm-off-packaging-pilot-20261127 --output <fresh-local-private-root> --request <new-32-hex> --parent <current-job> --node <current-node>
```

The second command creates a disabled source review and deterministic train members. For the enabled preparation use a new output directory and add `--publication <reviewed-source-commit>` and `--authorization <root-authority-json>`. Root must obtain exact current parent/node authority, confirm the endpoint is authorized for the six pinned members, and keep the node out of conflicting dispatch during admission. The authority JSON contains exactly `request`, `parent`, `node`, `publication`, `input_manifest_SHA256`, `CPU_GO:true`, `GPU_GO:false`, `training_GO:false`, `Model_GO:false`, and `data_on_target_authorized:true`. The original parent shape is 40 nodes and 320 DCUs. A changed parent shape requires reviewing the inherited controller admission first.

## Target entry

Root may install the exact six train members into a fresh private staging directory and atomically expose it as `R/incoming/<label>/train`, where `R` is the existing target checkout and `label` is printed by the builder. Holders of an approved stage channel should enforce exclusive writes and check the six-member readback before exposing the path. The installed train directory is mounted read-only as `/off-train`. The bound carrier's default packet limit is 560,000 bytes. The private train payload therefore requires an approved multipart or explicit file staging route. The controller source payload is separate and never carries train bytes.

```text
python3 -I -S -B off_inputs.py --root <canonical-absolute-R>/incoming/<label>/train --input-sha 9a1338a008b91e2524873e12f49517889a23af4a9869232d5fa1c6103a0e60d4
python -B code/bw1000/training-minicpm-off-native6-CPU-v1/prepare.py --payload-from <enabled-local-private-root>/src --source-manifest <generated-manifest-sha> --controller-payload <new-private-file>
```

Execute the payload file once as Python on the Slurm login host through the existing approved stdlib carrier. It checks the exact configured request, file identities, parent ownership and both active step views before one `srun --exact --gres=none` with 8 CPU and 12288 MiB. The source packet permits a 45-second admission freshness window from generation to the actual srun gate. Whole epoch is 300 seconds, Slurm budget at most 240 seconds, worker budget at most 210 seconds. A failed or unknown send requires a read-only reconciliation of the original results path and accounting label.

Run the independent reader on the login host with the generated source manifest SHA and exact request.

```text
python3 -I -S -B <canonical-absolute-R>/results/<label>/src/independent.py --request <new-32-hex> --manifest <generated-manifest-sha>
```

The local regression executes real input preparation and staged readback, checks exact native-call preservation in the derived source, and tests zero-GRES accounting predicates. Target staging, native calls, Slurm and GPU counts from local verification are all zero.
