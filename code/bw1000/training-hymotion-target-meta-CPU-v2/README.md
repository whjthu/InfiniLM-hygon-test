# Native HY-Motion CPU meta import boundary

This distinct source claim isolates the actual CPU179 FlashAttention shared-object import abort. Only the CPU meta checker excludes optional flash_attn imports; the original native attention/training backend remains byte-identical. Exact original A100 meta schema is required. No weights,cache tensor bodies,pipeline,forward/backward,training orGPU. One epoch300s/CPUwork220s;original request/intent/terminal accounting is reconciled without replay. The unresolved60vs128-frame question remains closed to training.
