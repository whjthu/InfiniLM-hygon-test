# MiniCPM OFF checkpoint-2 paired 14-question pilot

The target input directory is the original approved read-only `R/incoming/training-minicpm-off-eval14-30f70022a75f`. It contains only `prompts.answerless.jsonl` and the 11 JPEG names cited by its 14 rows. Each prompt has exactly `slot,group,image,image_SHA256,question`. Prompt SHA-256 is `30f70022a75ffba0b0b2088619f06083ab3423a27af72e7bbc85718b291039af` and its extent is 5055 bytes. No colocated manifest is needed. The seven groups and 14 original slots are frozen. The worker has no gold input.

## Install and admit once

1. Verify the source publication and stage on the approved login host from the published checkout.

```text
python3 -B host.py install --request <fresh-32-hex> --publication <full-commit> --source-manifest <manifest-sha> --cp-manifest-sha 11207b254a8c417d0729927500ecd1c23ed80462f16dfa925f3d4b2ec019fb95
```

The install verifies the existing answerless directory in place, all ten original model file SHAs and the original 14-member CP2 inventory and two actual adapter files. It reads the original target `independent-reload-cp2/result.json` (SHA-256 `03bbd7fe6047f6db2e93d4987fa1fa67f5d2130ae4fdb1e0148f4b58f0650220`) and live exact `1103606.4486` UTC accounting. The private archive is unnecessary on target. The target worker independently verifies all ten current model file SHAs and both CP2 SHAs with stable file descriptors, and compares inode, size, mtime and ctime across mounts. It copies only source into `R/incoming/<label>` and emits the `STAGE.json` pin. A partial stage remains for reconciliation.

If using a source-only target scaffold, include the exact checkout-relative dependencies `code/bw1000/training-domain-pilot-v1/internvl_gqa_miopen_home.py` (2388 B, SHA-256 `8bb74a51d200842edbdeb3cdb93e5f5264eeefa4efab41e985de61ecea522b61`), `code/bw1000/training-minicpm-usgs-native88-CPU-v1/providers.py` (3427 B, SHA-256 `4196a19e4f3fae4eb399ef76f752bbd4f009267abaebd4379237d71373951479`), and `code/bw1000/training-minicpm-final-tail-r2/MODEL_FILES.json` (1776 B, SHA-256 `ddcb8756dc66474f348bd81bb1730874ba78af8b46567e6a99596cedf5594ae6`). No training or prediction bodies are included.

2. Root supplies a single fresh authority record for a real owned node with an empty current numbered-step view, independent clean KFD/FD/physical evidence, resolved predecessor and exclusion from ordinary dispatch. The private authority fields are `request,parent,node,stage_SHA256,root_GO:true,ordinary_excluded:true,physical_clean:true,prior_holder_reconciled:true,observed_epoch,expires_epoch,boot,pci`. The lifetime is at most 60 seconds, and `pci` contains the eight actual canonical device identities. The CLI performs its fresh dual scheduler checks before it can record intent.

```text
python3 -B R/incoming/<label>/host.py preflight --request <id> --stage-sha <sha> --parent <parent> --node <node> --grant <private-file> --grant-sha <sha>
python3 -B R/incoming/<label>/host.py start --request <id> --stage-sha <sha> --parent <parent> --node <node> --grant <private-file> --grant-sha <sha>
python3 -B R/incoming/<label>/host.py terminal --request <id> --parent <parent>
```

Use the read-only preflight when diagnosing. The single start rechecks it itself. Keep root custody through the numbered step appearing. The start persists exclusive scope and intent and requests one DCU, 8 CPUs, 64000 MiB and 15 minutes. Unknown launch return requires status inspection of the original accounting label and path, with no resend. Root obtains independent post-device clearance and does not release or score on the worker output alone.

## Worker and scoring

The same process and original base weights generate base 14 then checkpoint-2 14 with Swift TransformersEngine and MiniCPM-V-4.6 native image template. The policy is BF16/SDPA, input cap 4096, max pixels 262144, no sampling, one beam, 32 new tokens. Both arms record exact processed-image/input fingerprints for every successful slot. The base arm sees a naked base model, and the adapted arm proves the registered CP2 adapter and exact CPU-comparable named tensors (372, 10,822,656 elements, SHA-256 `81b448fd3b6dcaad1de9db89aabbd419be5b9dafc610d53f611ba08aea61c939`). The load and native generation checks use original pinned Swift inference engine, wheels, common digest helpers and a fresh private MIOpen home. Runtime risk remains until the first actual native MiniCPM GPU call. The prior validated GPU Swift inference provider used InternVL, while MiniCPM's operator/template has prior native training and CPU evidence.

The worker writes `base.jsonl` and `adapter.jsonl` under the request's private `predictions` directory. Each reported row contains exactly `slot,answer,finish_reason`. The manifest seals both files and the fixed 28-slot ledger, which retains attempted errors and missing slots. A confirmed system error leaves remaining slots missing. The host terminal CLI checks hashes and independently reads exact UTC accounting without printing predictions. Only after complete sealed arms and independent clean post should an authorized local operator run the already-frozen `training-minicpm-off-eval14-v1/score.py` against exact file pins and local gold.
