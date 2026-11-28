# OneVision visual-input cache fix and two-forward CPU witness

The authenticated source audit is pinned at `5de9d05b399c906e5a587537096fb90a8c26897112e4d784854d317ce3065c74`. The installed Transformers 5.5 generation parent prepares and slices the effective query. The original OneVision multimodal forward at lines 1668–1675 already derives cache positions from actual written cache length and the effective embedding width when its input is None. It passes those positions to the text model and eager causal mask.

The repair is scoped to the single OneVision generation method on the loaded model. It preserves the original decision for a non-None position. For a None position in the qualified batch-one eager dynamic-cache route, it keeps visual inputs only when the real cache length is zero, the parent marks the first iteration, and the prepared query has more than one token. It drops vision on the subsequent one-token query after cache fill. The original forward receives the same None value and performs its existing fallback. There is no generated cache tensor, synthetic cache, or additional query slice. Unknown geometry is a failure.

## One new zero-GRES invocation

Use the same .5225 guest image, runtime and exact read-only mounts. Bind the frozen diagnostic generation-boundary source and v3 native package separately, plus the already-exposed single CLEVR train PNG and the original read-only OneVision model and receipt. Use 6 CPU and 64 GiB, no device attach, private `/tmp`, and a private mode-0700 output parent. The worker requires all three GPU visibility variables set to `-1`, the original offline environment, and actual numeric zero-GRES Slurm identity. Allow the outer deadline to exceed the worker's 520-second default enough for independent terminal reconciliation.

```sh
/usr/bin/python3 -B /repair/run_cpu.py \
  --boundary-source /original-5225-boundary --native-source /original-native-v3 \
  --diagnostic-image /exposed/CLEVR_train_014344.png \
  --diagnostic-image-sha e0e9e1c96951283e0ce9db9bf220499301a6d7b147dd1221793157b8b8f0912b \
  --model /llava-base/model --model-receipt /llava-base/receipt.json \
  --output /diag-output/once --request "$request" --budget 520
```

The same native Swift greedy invocation runs one actual prefill and one actual decode. The witness verifies the visual input is present only during prefill, real cache advances `0→N→N+1`, and both returned logits are finite. It verifies that the actual decode input token is the prefill greedy argmax. Tokens are represented by digest in the receipt. It stops after the second forward returns, before further sampling. A positively observed first-token EOS is recorded as decode coverage missing. The program opens no Val16, adapter or answer files and emits no prediction ledger.

`test_local.py` uses the exact published source readback to compile the old and corrected prepare methods, verifies original route binding and fixtures for position preservation, invalid geometry, two returns, and early EOS. It does not qualify target arithmetic, device behavior or held-out output.
