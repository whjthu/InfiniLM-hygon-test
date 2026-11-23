"""Observe target Torch's configuration setter and getter on CPU-only Slurm."""

from hashlib import sha256
import inspect
import json
import os
import socket
import warnings

import torch


cuda = torch.backends.cuda
choose = cuda.preferred_blas_library
implementation = inspect.getsource(choose)
before = choose()
rows = []
for label in ("default", "cublas", "cublaslt", "default"):
    data = {"requested": label, "enum": str(cuda._BlasBackends[label]), "before": str(choose())}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            returned = choose(label)
            data.update(returned=str(returned), returned_equals_enum=returned == cuda._BlasBackends[label])
        except Exception as error:
            data.update(error_type=type(error).__name__, warning_of_no_device="no" in str(error).lower()
                        and "device" in str(error).lower())
        after = choose()
        native = getattr(torch._C, "_get_blas_preferred_backend", None)
        data.update(after=str(after), getter_equals_enum=after == cuda._BlasBackends[label],
                    native_getter=str(native()) if native is not None else None,
                    warnings=[str(w.message)[:230] for w in caught])
    rows.append(data)
choose(before)
assert choose() == before and not torch.cuda.is_initialized()
print(json.dumps({
    "passed": True, "step": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
    "node": socket.gethostname().split(".", 1)[0], "torch": str(torch.__version__), "HIP": torch.version.hip,
    "preference_module_SHA256": sha256(implementation.encode()).hexdigest(),
    "setter_return_statements": [line.strip() for line in implementation.splitlines()
                                 if line.strip().startswith(("return ", "torch._C._set_blas"))],
    "initial": str(before), "restored": str(choose()), "rows": rows, "GPU_initialized": False,
    "tests_actual_kernel_dispatch": False,
}, allow_nan=False))
