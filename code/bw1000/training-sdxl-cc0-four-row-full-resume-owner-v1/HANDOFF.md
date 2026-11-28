# SDXL four-row native continuity and CP20 recovery

The controller trains from the existing .2766 ImageFolder on parent `1194900`, node `f13r4n03`. The input has four 256×256 RGB rows (tool chest original and side, painted stool original and side) and one metadata file. The owner checks all five unique members against the original .2766 archive before staging and again at admission. The native guest also decodes and hashes all four RGB rows. Captions are checked against .2766’s raw UTF-8 hashes and separately expressed in the trainer’s JSON-string hash convention. The held-out cassette player, electric kettle, and telephone wall clock are excluded by the five-file inventory.

The launch creates a fresh run with 25 native updates and saves CP10, CP20, and CP25. After the first process completes, it opens and validates the new CP20 native state at epoch 4, consumed 4. The second process uses the unmodified native checkpoint sampler restoration and runs steps 21–25 only. Update coordinates at 19, 20, 21, 24, and 25 are `(4,2)`, `(4,3)`, `(5,0)`, `(5,3)`, and `(6,0)`. All fully traversed epochs require exactly one use of each sample ID. Paired actual sample/collated tensor evidence, loss, gradient norms, and the final full state must match. The original .2675 run is historical method evidence, and its checkpoints cannot enter the new command.

## Stage on HOST

Use the original .2675 stage dependencies and the new frozen owner and native package. Stage reads the original retained input in place. It never starts a job. Treat a stage whose final receipt is absent as partial and reconcile it before using another request.

```sh
/usr/local/python3.12/bin/python3 -I -S -B /approved/four-owner/host.py stage \
  --request "$request" --manifest-sha 1b530fa137f9a0100eff0a9497633317ab9bb24851ffdf627c2f301ada9aa1e1 \
  --base-host /approved/first-update-owner-v3/host.py \
  --common /approved/llava-val16-owner-v1/host.py \
  --support /approved/llava-first2-owner-v1 \
  --pilot /approved/fleet-dcu-standby-control-node-pilot-v2/pilot.py \
  --legacy /approved/fleet-dcu-standby-f13r4n03-v1/shared_standby.py \
  --runtime-import-proof /approved/n03-2666-runtime.json \
  --tracker-proof /approved/n03-2667-tracker.json \
  --runtime-terminal-proof /approved/n03-2666-2667-terminal.json \
  --full-source /approved/four-native \
  --runtime-proof /approved/2672-normalized-rootfs-and-provider.json --runtime-proof-sha "$runtime_sha" \
  --four-proof /approved/specialized-sdxl-2766-train-only-two-families-four-native-imagefolder-rows.json
```

The native package manifest is `19843aa3c7c74267088759f43edd19980e77abdbf5b9c935a471e2a7905a34c7`. The normalized runtime proof is the `rootfs` and `provider` subobjects from the independently archived `specialized-sdxl-2672-dtk-rootfs-content-and-native-cp20-provider-zero-gres.json` (SHA-256 `23dcd817a886e910a96a8ae26a0a7cd417aeb920562be504ad836a15c0311030`), retaining the exact original image digest. Stage rechecks the DTK image directory identity and original software providers. The guest uses the original SDXL 18-file verified official pipeline, the stock b3e85543 trainer, the 1965014b common backend, the 3392f0ea helper, and the original offline DTK runtime. Their three derived provider digests are unchanged from .2675. The study uses its original BF16 AdamW and LoRA recipe, workers 0, microbatch 1, gradient accumulation 1, constant learning rate 1e-4, and maximum 25 steps.

The final receipt is `continuity-stage.json`. Use its returned SHA-256 in the distinct request’s ordinary root grant. The unchanged grant fields are `request, parent, node, stage_SHA256, epoch, deadline, port, boot, pci, root_GO, ordinary_excluded, fresh_physical_SHA256`. The root must independently verify fresh node placement, current step exclusion, the eight-PCI roster and boot, KFD/FD/port clearance and roles. Start enters the original shared standby lock and performs separate actual zero-GRES pre/post. Protected legacy .189 handling remains the exact old guard. The one GPU request is 8 CPU, 64000 MiB, 1 DCU, up to 30 minutes. Pre/post each request 2 CPU, 2048 MiB, 0 GRES, up to one minute.

```sh
R=/public/home/aclsylqidf/hygon-inference-refactor-20261014
prefix=$(printf '%.12s' "$request")
stage="$R/incoming/sdxl-cc0-four-full25-cp20-$prefix-stage"
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" start \
  --request "$request" --auth /approved/fresh-root-authority.json --auth-sha "$auth_sha"
/usr/local/python3.12/bin/python3 -I -S -B "$stage/host.py" terminal --request "$request"
```

The independent HOST reader rehashes all four checkpoint inventories, verifies the CP20 cursor evidence against the original CP20 manifest and its four actual epoch-4 sample IDs, and compares both update ledgers. Reading native CP20 tensor state, cursor, and hashes happens in the exact Linux guest. HOST reads the digest-bound cursor summary without deserializing tensors. Training and post cleanup are separately reported. A successful run is a continuity result with zero formal delivery credit.

## Local validation

Run `python -B test_local.py` in each source directory. The owner test uses frozen source plus fixture-only scheduler/asset availability and opens the real .2766 public receipt. The native test runs the real source derivation and native CLI parser, the real derived sampler cursor validator, four synthetic Pillow RGB fixture members, wrong membership, tampered bytes, incorrect epoch/step, duplicate occurrence, interrupted denominator and paired loss drift. The .2766 target media and a fresh GPU owner were never accessed by these local tests.
