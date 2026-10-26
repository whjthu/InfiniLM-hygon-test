# DeepSeek owner-prefill diagnostic r9165

One unchanged DSO edf1e808 weights load. Four real first prompts cap1: OFF warm, OFF before, arm1 exactly one packed PF then immediate read, OFF after. Main workload: first32 distinct immutable fresh64, B32 compact EOS, cap512, ownerOFF.

Read must contain all8 devices ×43 layers,32 markers and16 spans each. The16 spans are nested and cannot be added. Stream event elapsed includes launch gaps/dependency or collective waits and is not pure busy time. arm/read/Python validation and bracketing native-forward times remain separate. PackedPF counters, finite/currentargmax, readonly input and canary checks stay enabled. No rawactivation trace.

Only CPU mocked generation is validated here; no GPU task submitted. Reuse bounded Slurm CPUkeeper with clean child environment, actual64CPU nogres affinity beforeGPU, watchready/freshKFD/normaldestroy/post. Nodeparent910374 a02r4n05,8cards,c64,128000MiB,3600hard/3500inner. Root must publish source and claim before unique GPU launch.
