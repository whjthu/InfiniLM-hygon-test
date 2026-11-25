# Qwen3-VL street-scene CPU input check

The owner stages the existing private 5,355,621-byte ZIP using the existing 2,980-byte private transfer receipt. It independently checks all 22 entries, including the exact three original checker files, then installs the dataset into a new request namespace. The dataset contains 41 original training rows across 16 images, with 26 TextVQA and 15 OCR rows. Stage output contains only scalar evidence. Keep the archive, transfer receipt, claims, worker logs and inputs in the existing private channel.

The operator verifies ordinary publication of this owner and the original checker dependencies, then supplies the owner at an absolute target HOST path. Send only the exact two private files required for staging. The installed destination is `results/training-qwen-textvqa-native-input-CPU-<request>` beneath the existing project root.

```sh
/usr/local/python3.12/bin/python3 -I -B "$OWNER" stage --request "$REQUEST" --zip "$PRIVATE_ZIP" --receipt "$PRIVATE_RECEIPT"
```

After the real placement is known, create a private claim keyed to the actual published source commit. The supplied claim contains `request`, `publication`, `parent`, `node`, `owner_SHA256`, `CPU_GO=true`, `GPU_GO=false`, `exclusive_node=true`, `replay_allowed=false`, `cpus=8`, `memory_MiB=12288`, `allocated_DCU=0`, `max_deadline_s=480`, `planned=41`, `zip_SHA256` and `dataset_SHA256` from the owner. The private fresh authority contains the same request, source publication, parent and node, plus `exclusive_cpu_admit=true`, `GPU=0`, `coordinator_proof_SHA256` and `observed_epoch`. The operator independently verifies that coordinator proof, keeps the node reserved in the active coordinator, and supplies the file digests from the actual approved claim and authority. Freshness runs from the latter observation until durable intent, with a 60-second maximum.

```sh
/usr/local/python3.12/bin/python3 -I -B "$OWNER" start --request "$REQUEST" --publication "$PUBLICATION" --claim "$CLAIM" --claim-sha "$CLAIM_SHA256" --authority "$AUTHORITY" --authority-sha "$AUTHORITY_SHA256"
/usr/local/python3.12/bin/python3 -I -B "$OWNER" terminal --request "$REQUEST"
```

The HOST checks parent ownership and per-node resources, consults both Slurm step views, and requires that the parent budget covers the full bound. It persists one intent before `srun --gres=none`. The isolated original Swift/DTK launcher runs one processor and collator check with no real model parameters and no optimizer updates. The independent terminal command combines exact nine-column accounting with the numeric worker marker and original native report. After intent, any unknown transport outcome goes to terminal reconciliation. Keep the planned denominator of 41 on incomplete or missing results.

Run `check_local.py --zip … --receipt …` with the frozen private files to verify the actual stage and original frozen input loader without scheduler access. The local test captures the single launch and exercises terminal validation with fixtures.
