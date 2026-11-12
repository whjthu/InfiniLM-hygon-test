# Original Qwen CPU structural qualification

One new original-A100 CPU process constructs the complete Qwen3-VL architecture on meta tensors, uses actual Swift get_model_processor(load_model=False), actual Swift target expansion and prepare_adapter, actual PEFT, and actual Trainer arguments. Gradient checkpointing is enabled as required by the original recipe. It checks the complete trainable manifest, model.visual frozen boundary, source providers and 15 deliberately invalid structural boundaries. Eight shared stdlib checks are also run once in the original environment.

The seven shared source files come exactly from framework e2d7ca2cfebe9ab76d2d8ba8afc063665fada070 (338-file framework); only this declared subset is installed. The original config, 750-key index and three implementation files must still match their published hashes. No model weight values, image samples, forward/backward, optimizer update, trainable initialization or GPU use are authorized by this scope.

One epoch starts before installation: whole 300s, CPU signal deadline epoch+260, supervisor work stop epoch+270, TERM/KILL cleanup 5s each, four CPU-affinity cores, two Torch threads and 8GiB aggregate RSS limit. The foreground supervisor owns a new process group and records terminal status plus cleanup. It is not a persistent SSH service or poller.

A successful result qualifies the original structural binding only. First-batch preprocessing, full-output capture-size qualification, native GPU reference, cross-platform numerics, 25-step training/recovery, formal performance and held-out effects remain separate pending gates. Existing initial-value or private-input export denials are unaffected.
