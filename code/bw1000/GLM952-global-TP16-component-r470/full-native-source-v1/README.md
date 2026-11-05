# Private GLM952 TP16PP1 fullNative source candidate

This package implements a separate typed factory with two16rank comms, globalrank=node*8+localCUDAdevice, NoV4attentionheads/2indexheads, PP1all78layers and C1full32768. Original public TP8PP2 source/default and its installed952 library are unchanged.

Routed experts use physical/logical I128, globaloffset=rank*128 and replicated16scale blocks; the private helper is the original raw_r19 FP8/F32tree, rather than padded256 or the row2 route. All TP16 routed widths use this reference leaf. Current TP8 grouped hybrid prefill acceleration is not inherited, so prefill may be much slower. Changed TP reduction partition and BF16 partial rounding need actual numeric/quality evidence. NoModel speed/fit/quality is established.

CPU scope is exactly8HOST+5GPU TUs (including4fresh originalhelpers whose oldMDs are unavailable),18original objects reused,31objects linked. All13fresh-M closures must match1546exact dependency pins plus these exact source bytes before any compile. CUDA compiler code generation runs on CPU; no generated kernel, SDK, Runtime, Common or Native DSO executes. Existing genuine804/952receipt and current Common16910/70dynamic identities are bound. The new typed caller is exercised with explicit CFUNCTYPE recording leaves only.

A future device claim must separately close fullcheckpointloader/globalshapes, constructor/cache/allocator floor, everyTP16NoV/indexer/FFN numeric path, all16owners/rootoutput/normalcleanup. Completed876 establishes two comm16 qualification only; it supplies noModel cache/forward or fit qualification. GPU_GO remains false.
