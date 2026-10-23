# GLM ROWS32 one-load experiment snapshot

Final unique entry: `launch_GLM_rows32_Model_r5739.py`; source/memory plan is r5738 and private native CPU build is r5734. Older entries are retained as provenance; do not execute them in parallel or replay an unknown execution.

The r5739 change only keeps the terminal observer following the same actual Slurm handle after transient accounting-read failures. Numeric probe, native DSO and pinned dependencies are unchanged.

TP8/PP2,16cards,B32/M8,32distinct actual512token chat-template inputs,max128output. V fixed ON/T fixed OFF; resident policy is the single paired variable. Same-session OFF complete tokens/EOS are the regression reference for this new geometry. B32/M16 historical gold is not reused. Fresh all-owner memory admission selects fixed stage masks; no full40layer fit assumption.

One necessary load: first I512 O/R/R/O, then short historical anchor and B32 QA, then a B8/K4 representative profile. No throughput/adoption claim until actual normal exit, complete tokens/EOS, independent audit and post-KFD check.

Build sources are exact upload-name snapshots; the CPU build script maps them to their original private header/source names. Weights, raw activations, DSOs and SSH credentials are not included.
