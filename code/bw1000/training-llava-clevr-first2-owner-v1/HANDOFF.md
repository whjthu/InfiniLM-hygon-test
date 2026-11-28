# T04 native first-two once owner

Use the existing source-only staged T04 tree and the original train-only view. This owner consumes the actual CPU tensor result from .4878, the original recovered 18-member base model, and the three exact shared physical/accounting/owned-process helpers from the qualified production path.

The root authority for each unique request contains `request`, `parent=1103606`, `node=f13r1n14`, `label=training-llava-clevr-first2-<first12>`, `port`, `epoch`, `deadline`, `GPU_GO=true`, `node_exclusive_to_request=true`, `release_SHA256`, `owner_manifest_SHA256`, `science_manifest_SHA256`, `TENSORS_SHA256` and the exact `vision_root` in owner.py. The epoch starts after the prior holder has released. The owner accepts it for up to 30 seconds before the first step. The run name and all Slurm labels are single-use.

Supply a separate release document bound to the same request. Its fields are `request`, `mode` (NATURAL or ORDERLY_ROOT_HANDOFF), `parent`, `node`, `holder_step`, `holder_label`, `accounting` (original exact UTC nine columns), `worker_SHA256`, `plan_SHA256`, `independent_post_SHA256`, `worker_returned=true` and `independent_physical_clean=true`. Orderly handoff also requires `exact_root_handoff_confirmed=true`. Root must bind the exact owned holder and protect the node from outside refill until post. The entrant verifies live terminal accounting again under the original shared fleet lock. A release intent or active holder does not pass. No standby signal is sent by this source.

```sh
/usr/local/python3.12/bin/python3 -I -S -B "$OWNER_STAGE/owner.py" launch \
  --run "$R/results/$LABEL" --auth "$AUTH" --auth-sha "$AUTH_SHA" --release "$RELEASE"
```

For a root-owned orderly handoff, use `--inherited-lock-fd FD` and pass the exact already-held open file description through exec or `pass_fds`. The owner checks the descriptor's inode and device against the original fleet lock, then acquires the same exclusive lock. The caller must not concurrently manipulate the lock. The owner unlocks after GPU dispatch has an exact allocated numeric step, allowing other nodes to refill during training.

Pre and post are separate two-CPU, 2-GiB, zero-GRES numbered HOST steps. The compute gate checks the actual cgroup, hostname, allocation, name and TRES. Pre checks all eight PCI addresses, the boot identity, KFD and same-UID device descriptors, and the dedicated port. A fresh allocation on the same boot must match that PCI set before the owner writes the training authority and creates one private TorchElastic process group. The guest uses the original DTK image, read-only model/assets/source, private HOME and cache roots. A signal or timeout stops only recorded request-owned process identities. The post process must independently verify matching boot/PCI, no KFD/device descriptors, port clear and no live recorded children.

The n14 HOST step uses the qualified Python 3.12 path and binds `/usr/local/hyhal` read-only to guest `/opt/hyhal`, as in the actual .4794 run. Pre and GPU compute admission require that HOST source to be present. The DTK guest keeps `/usr/bin/python3`.

After GPU terminal evidence, the owner attempts post even if training failed. It reports release and science as separate outcomes. Unknown sends remain single-use. Read existing receipts and exact labels using the `observe` mode with the same `--run` and `--auth-sha`. No mode replays a GPU intent. Retain the bounded rank files and logs for incomplete scientific or cleanup outcomes. A per-node /tmp cache tree is request-private and is retained with its ownership parent.
