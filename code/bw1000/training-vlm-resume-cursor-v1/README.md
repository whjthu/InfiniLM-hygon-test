# VLM epoch-boundary resume inspection

The shared framework tool `tools/inspect_vlm_resume.py` reads SHA-pinned Trainer state and saved arguments, then checks an epoch-boundary cursor without importing Torch or loading model state. It rejects a restarted schedule, mismatched batch/accumulation, weights-only resume, skipped data, and unsupported partial epochs.

Seven CPU tests passed locally and on the Hygon HOST. The actual checkpoint20 JSON implies five remaining updates under the unchanged 25-step schedule, conditional on verifying the proposed loader length and topology at runtime. Original fresh-run callbacks require a separate resume witness. No GPU launch, tensor restore, following-input equality, numerical tolerance, performance or heldout qualification is established.
