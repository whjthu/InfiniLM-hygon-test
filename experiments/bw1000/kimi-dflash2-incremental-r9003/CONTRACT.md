# Default-OFF incremental DFlash2 worker

The worker is `dflash2_incremental_worker_r9003.IncrementalDFlash2`.
Construct with the pinned five-layer draft module, an external LM-head callable,
`enabled=True`, and the target's explicit `eos_ids`. Default construction fails
closed. This is a single-owner, batch-one PyTorch reference worker, not an
InfiniLM CAPI or qualified BW1000 GPU backend.

* `append_context(features5tap, positions)` accepts five [1,T,H] target hidden
  tensors in tap order [19,37,66,78,90], or their concatenation. Positions are
  contiguous absolute token positions, starting at zero. Only actual target
  features are appended. No draft hidden states may be passed as target features.
* `speculate(anchor)` computes eight query rows: committed anchor plus seven mask
  tokens. It returns all seven unverified candidate tokens and query hidden rows.
  A candidate EOS is not treated as committed. A committed anchor in `eos_ids`
  returns an empty terminal result without running the query block.
* `crop_context(acceptedprefix)` takes the absolute number of retained target
  context tokens, not the number of accepted draft candidates. Only accepted
  target features supplied by the target verifier can subsequently be appended.
* `close()` drops all worker references to KV, model, head, projection and norm
  modules; caller-owned model/head references remain caller-owned.

Every creator layer uses the same projected target context; context does not
flow through the five draft decoder layers. Cache K/V is therefore immutable
until append/crop. Query K/V is ephemeral and never committed. Four layers use
the original bidirectional abs(position difference)<4096 mask and one full
attention layer. Original YaRN, RMSNorm, both GroupedConv operations, MLA, MLP,
final norm and selector formulas are reused. The pinned config has fc_norm=false
and input_embedding_scale=1; other architectures are outside this packet.

Original GroupedConv kernel2 resets on each eight-query block. Its previous-row
dependency is evaluated inside that block by the unchanged creator class. There
is no recurrent convolution tail across speculative blocks or target context.
Diagnostic dynamic-kernel tails are cleared on each speculate/crop/close.

The simple cache retains expanded K/V for all context tokens in all layers.
At actual BF16 dimensions it is 204800 bytes/context-token (about 0.782 GiB for
4097 tokens), excluding temporary query/concatenation buffers and allocator
overhead. Long-context fit is not claimed; a future GPU runner needs a measured
working-set admission bound. append/crop currently allocate concatenated/cloned
buffers; this packet establishes semantics, not incremental performance.

CPU qualification uses synthetic F32 tiny weights and CPU SDPA in place of the
creator flex GPU backend. No trained-weight acceptance, GPU latency, target KDA
state lifecycle or online speculative speedup is established here. LM-head
values, target embedding compatibility and BF16 GPU behavior remain untested.
The one authorized remote checkpoint download failed at DNS before payload and
was not retried. No weights were downloaded to the local workspace.
