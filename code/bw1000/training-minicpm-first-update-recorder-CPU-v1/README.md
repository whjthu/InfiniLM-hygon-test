# First-update recorder CPU qualification

Exact source snapshot of unified framework72d42dc, including0d0899d callback order.
One unique CPU-only Slurm step on T05 f11r4n09 in parent1087656;4CPU/8192MiB.
One240s origin epoch, inside150s,Slurm180s,outer200s,ownedTERM5/KILL5.
Reuse the previously successful full target DTK/Swift/Transformers environment.
Seventeen tests: initializer4, ordered launcher7, bound-provider inventory2,
real-PEFT/AdamW artifact core4. Tiny synthetic CPU tensors only; no original
model, weights, Native runtime or device. Importing the production callbacks
must not initialize CUDA; their actual model on_train_begin is not invoked.
Original GPU model/math, numerical alignment, recovery, performance and effects
remain unqualified. Do not replay this label or launcher on observation timeout.
