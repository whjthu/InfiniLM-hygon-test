# Original MiniCPM eight-card HIP native25 diagnostic

Use the installed unified framework668bfe9, independently qualified by CPU130.
Only T05 f11r4n09 is used:8DCU/64SlurmCPU/224000MiB from parent1087656.
The companion f11r4n10 remains the pool for later fresh-process restore/effects.
The original8cardBF16 batch1/rank,accumulation4,effective32,max4096,SDPA,
GCoff,rank16/alpha32/dropout.05,AdamW/cosine/warmup5/native25 are unchanged.
Source/model10files/data67files and8originalmask manifests are verified before
actual dispatch; each complete mask packet is SHAchecked when actually read.
Initial LoRA43466515B/d382 and named9ac are restored before both recorders.
First2updates/8microforwards per rank retain all200 full private tensor files;
64originalmask packets are replayed while advancing native destination RNG.

One original epoch:preflight180/native7200/inside7260/GPU7380/whole7800s.
Slurm minutes are rounded down to the same epoch,not renewed at GPU admission.
Native process ownership uses exact output/receipt identity andPID/startticks.
The finite controller supervises exactly preCPU/GPU/postCPU steps and reuses the
actual qualified bounded terminal-accounting helper; RUNNING/missing is not
terminal. FullPCI/KFD/boot/port/ownedPID checks gate admission and post.
A detached per-job controller is finite; noSSHservice or polling service exists.
Unknown original status never authorizes another invocation of this label.

This includes initialization, mask replay, CPUcopies,full artifact serialization,
provider file hashes and completecheckpointIO. It is not formal performance.
A100 native reference drift remains UNKNOWN and no original numerical tolerance
has been supplied. Neither source readiness nor producer counts qualify model
math, numerical alignment, checkpoint restoration, performance or held-out
TextVQA/COCO. Independent artifact/numerical consumption follows actual terminal.
Raw tensors, prompts,mask payloads and checkpoints stay private on experiment
storage. Only necessary source and bounded summaries enter the repository.
