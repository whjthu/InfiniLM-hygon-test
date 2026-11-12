# Original HY-Motion Transformer meta/LoRA CPU qualification

Data-independent exact native Transformer construction only. The60/128frame question remains pending; no training is authorized by this CPU claim. Current unified331backend helpers are AST-checked against the original A100 backend before use. All model parameters must stay meta; base is frozen,LoRA rank4/alpha8FP32,canonical parameter names/shapes must survive activation-checkpoint wrapping,AdamW must own exactly those parameters with empty state.

Source16Python hashes are checked before import and runtime hymotion modules must belong to that exact manifest. Source namespace packages are separately recorded. No repository revision inferred from missing .git. Optional textencoder/pipeline dependencies,real buffers/weights,data,device numerics and performance remain unqualified.

One240second epoch includes exact namespace/intent,one two-core-affinity CPU child,180second child alarm,external timeout at epoch+190,5second kill grace and ownedchild reap. The4GiBRSS check is observed after construction,not a scheduler memory reservation. Only public code/config is staged; no checkpoint/cache tensor transfer.
