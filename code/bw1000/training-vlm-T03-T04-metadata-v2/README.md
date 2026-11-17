# T03/T04 metadata staging v2

This is a new metadata relay scope, not a training driver or a weight transfer.
Root publishes the exact package and invokes relay_once.py once, with the actual
publication and SOURCE_FILES.json SHA. No command in preparation invokes SSH.
The dispatcher also git-shows the frozen published v1 independent terminal
evidence and requires the original source PID absent, actual receiver Python
exit1 before namespace/payload, and the combined zero-write proof. The original
transport UNKNOWN is retained alongside that independently reconciled terminal;
v1 is closed and never replayed. This separately registered v2 repeats32,327,445
logical source bytes, which must remain in the total cost denominator.

Only A100-27 ordinary SSH and the existing pinned Hygon helper are used. Source
and receiver each create their fixed exclusive namespace and intent with PID,
startticks, UID, boot ID, original epoch and deadline. Existing namespace rejects;
UNKNOWN requires original-request reconciliation and never authorizes replay.
The source reads 27 exact metadata files, eight 8-byte-prefix/header ranges, and
32 pinned small training JSON records. It reads zero shard tensor payload bytes
and zero initial.pt bytes. Source JSON records never enter the relay bundle.
The 36 fixed output members comprise 27 metadata files, eight header files and
one canonical initial-bindings.json containing two separate eight-rank sets.
The binding reports source JSON paths/bytes/SHA and initial file bytes/SHA/named
digest, with global25/two-captured-update checks. Same dimensions or empty-state
hashes cannot substitute for a case's initial values. File-SHA agreement and
named-digest agreement are reported separately, without reloading tensors.
Source fsyncs bundle.jsonframe.gz, initial-bindings.json, validation.json and its
own directory before its first binary stdout write. Failed validation/fsync
cannot return a durable success; later broken-pipe retains the source artifacts
and a failed endpoint outcome for original-request reconciliation.

Both endpoint processes use two CPU affinity cores, nice10 and RLIMIT_AS512MiB;
this is not a cgroup hard RSS guarantee. They import only stdlib. One root epoch
of300s covers source and receiver; endpoint SIGALRM and root wall/monotonic
remaining budgets shrink against that deadline. Root retains only the bounded
compressed bundle in memory, never local raw metadata or a raw bundle. Logs and
summaries use separate streams. No SSH service, poller, Slurm, Torch, SDK, model
load, key read, fallback host, cleanup of shared files or device ownership probe.

Exact read bytes:32,327,445 =32,018,497 metadata +184,656 header prefixes/headers
+124,292 original JSON. Derived binding <=65,536 bytes, member <=12MiB, index
<=65,536 bytes, outbound member payload <=32,268,689 bytes and uncompressed
framed bundle <=32,334,233 bytes. Compressed cap32MiB; decompression cap64MiB.
Source stats the complete shard's identity/size but never whole-hashes its body.
Header files use unbuffered Python fdopen. All read budgets and zero tensor
consumption statements describe application logical bytes. Kernel/filesystem
readahead and physical IO remain UNKNOWN. No measured physical-zero-IO claim.

Receiver strips all squeue fields and checks scontrol's numeric UserId against
os.getuid; printable queue/UID names are diagnostic only. Both parents must
still be RUNNING40, AllocTRES node40/DCU320, with80 unique hosts.640 is reserved
cards, not measured busy. Each endpoint records actual nice, getrusage(self)
CPU seconds, Linux maxrssKiB converted to bytes and endpoint wall. These are
process system costs; they neither measure GPU busy nor impose a cgroup RSS cap.

Receiver metadata-only success proves these staged metadata/header bytes and
the extracted source-record identities. It does not prove a complete target
model exists, target weight values, tensor-copy identity, cross-platform RNG or
dropout parity, first-step numerical alignment, training, performance or effect.
GPU_GO, Model_GO and full_weight_identity remain false after metadata success.

Local verification: verify_package.py and check_v2_CPU.py with the fixed local
Python. Only the new UID, durability failure, prior-terminal and binary-bootstrap
boundaries are exercised. File writes/fsync run for real locally; POSIX directory
fsync is a checked call boundary, not qualified on Windows. Root separately
consumes actual original endpoint intents/outcomes after a real invocation.
