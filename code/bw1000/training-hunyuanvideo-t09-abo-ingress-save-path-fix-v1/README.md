# T09 ABO ingress save-path correction

The actual V2 GPU process at `1103606.4543` reached the unified training CLI. The captured stack locates the same `PurePosixPath.open` error at the two calls to the writer in `domain_ingress.run_common`, during cache ingress and the Muon receipt in its `finally` block.

`contract.patch` converts the path-like argument to the platform's concrete `Path` before opening. The original exclusive create mode, serialization, flush and fsync apply. `contract.py` is the full corrected byte comparison against the frozen V2 contract (original SHA-256 `6d8c7f48ef7a652547a749396673fe1de98bad51e7de24498b536e7080777a2b`). Rebind any future production scope independently and use the one-line delta. The full comparison file contains the old V2 request identity for reproducibility.

```text
python -I -S -B -X utf8 code/bw1000/training-hunyuanvideo-t09-abo-ingress-save-path-fix-v1/check_local.py
```

The check uses exact frozen `domain_ingress.run_common`, unified training CLI, its sibling provider, the original launch bytes, and the real exclusive writer. The deterministic fixture stands in for the common CLI's downstream GPU backend and optimizer callback registration. It reproduces both original path failures, verifies a normal 25-hook witness and successful review, checks a first-forward failure propagates with an explicit 0 observed/25 missing close record, and verifies a duplicate leaves both original records byte-for-byte intact. The Windows fixture projects temporary-file stat metadata to file size while still checking source and cache SHA. Actual native updates and GPU invocations are both zero. This correction supplies source evidence only.
