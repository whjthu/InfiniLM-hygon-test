# Leased Dense row-shard complete-MoE component

One layer1 C1 TP16/PP1 component. Original full BF16 low-rank weights remain resident. OFF is original strong forward. ON uses original Dense provider on output-row weight views plus two original AllGather collectives. Original router, FP4 experts, routed reduction/norm, shared path and combine remain.

CPU912 qualifies static production source/link only. The same caller scope retains14 Tensor handles through the original full-forward completion and clears/frees them inside the primary timer; OFF has an empty symmetric scope. Original first-submit affinity policy remains unchanged. No r432 kernel is linked.

Six arms 0,1,0,1,1,0: first two warm, four formal. Max16 rank-local complete CPP wall, adjacent pairs and OFF endpoint drift; all outputs/cold/warm retained. Historical teacher input and PP1 padded-scale fixture do not establish current PP2 Model quality, capacity, performance or public-service qualification.

Worker admission is original epoch+540s, shell hard570s, GPU600s, keeper900s. Complete Native-session <=300s is a result-qualification guard after return, not an additional watchdog. No deadline reset/reload/retry. Raw BF16 outputs and logs remain R/L.
