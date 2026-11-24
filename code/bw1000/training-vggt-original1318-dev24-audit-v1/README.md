# VGGT .1318 camera head Dev24 provenance audit

The candidate is the continuous 25-step camera head from training step `1103606.1318`. Its original clean post is `1103606.1319`. The public checkpoint file SHA and camera state-tree SHA identify the training boundary. Exact Dev24 deduplication uses the exported camera head file SHA and native module SHA from the original continuous result.

## Local commands

```powershell
python -B -X utf8 code/bw1000/training-vggt-original1318-dev24-audit-v1/audit.py local --output local-only/vggt-1318-independent-absent.json
python -B -X utf8 code/bw1000/training-vggt-original1318-dev24-audit-v1/audit.py render-layout --output local-only/vggt-1318-independent-layout
```

The input audit reads the original eight training pair manifests and the original frozen Dev24 zip. The exact original scorer checks 22 shared RGB224 and pose inputs, 11 per Dev recording. The 24-window denominator includes Freiburg1 plant window 01 with `RGB_TIMESTAMP_GAP` and Freiburg2 dishes window 01 with `LOW_TRANSLATION`. Both arms retain both missing scoring slots. The training recordings are `freiburg3_structure_texture_near` (7 pairs) and `freiburg3_nostructure_texture_far` (1 pair). Neither occurs in Dev24.

The generated source layout uses the actual scorer-only portion of the pinned .1506 renderer, the canonical directed-pose math, the original independent auditor and its `2e-15` AUC roundoff gate. The candidate declaration and candidate mount argument select the .1318 continuous output. The original thresholds require candidate AUC@30 at least baseline, AUC@15 at least 95% of baseline and failures at most baseline. Scoring is development only.

## Future target readback

Run `audit.py target-read-only` from the independently verified source closure on the authorized shared host. It reads the original .1318 controller outcome, combined native result, continuous receipt and native result by their linked digests. It binds the original exported head and completed checkpoint paths, rechecks the actual original model/provider, reads Dev24 stage and RGB/pose inputs, and binds the earlier .1038 head through that run's hashed score intent and head authority. It stops when an earlier identical-input run has the same export file SHA or native module SHA. It also stops when there are no source-disjoint input pairs. A changed denominator is a scope error.

The target readback checks checkpoint and base weight extents. The eventual scorer streams and hashes the exported camera head and original model before loading and verifies the live camera module and frozen aggregator before and after scoring. Preserve the readback result and write the enclosed `head_authority` JSON alone to a private request-specific file. Pin those exact bytes. The target still needs fresh independent placement and terminal/cleanup evidence. Historical GPU placement is provenance only.

## Paired scorer and independent audit

Use the generated source layout in an isolated request directory and the already verified .1506 runtime. Bind the source tree, original Dev24 input directory, original .1318 result root, VGGT upstream and original base weights read-only. Give the scorer one separate private output directory. Its argument layout is:

```sh
python3 -B -u /source/tools/score_vggt_tum_cross_camera_dev.py \
  --inputs /dev24/inputs-private.json --inputs-sha 3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6 \
  --head-authority /proof/head-authority.json --head-authority-sha "$HEAD_AUTHORITY_SHA" \
  --candidate /candidate --upstream /vggt-source --weights /weights/model.pt --output /request/score
```

The scorer processes the same 22 ordered pairs through the base model and through the same model with only its camera head changed. The denominator is 24 per arm. Independently run the generated `tools/read_vggt_dev24_roundoff_audit.py` with the original reader path and SHA, inputs path and SHA, authority SHA, private progress JSONL and a new private result path, in that order. Then run `tools/dev24_decision_1318.py PRIVATE_AUDIT_JSON` to apply the fixed development thresholds.

This local delivery establishes zero target reads, native predictions, optimizer updates, GPU calls or scheduler submissions. Exact cross-run deduplication remains open until the .1318 export and the .1038 scored head authority can be read on the target.
