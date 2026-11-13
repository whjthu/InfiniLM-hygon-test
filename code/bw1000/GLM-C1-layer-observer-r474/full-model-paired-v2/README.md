# GLM-5.3 paired screen with common progress logging

Both arms use this worker and identical append-only progress logging. They retain the original 33 requests, context 32768, output cap 4096, and 8 GiB memory floor. The only selected arm difference is the frozen Native library. Reuse the original checks for terminal accounting, device ownership, and request lifecycle.

The original failed screen provides a sizing scenario. The slowest observed completed-request sum plus five unresolved requests at that rank's slowest capped-request wall time is 41613.74 seconds. Adding the existing 10800-second loading ceiling gives 52413.74 seconds. Set the application limit to 57000 seconds without claiming a bound on unobserved work or subtracting the CPU callback measurement. Use 57300 seconds for the container wrapper, 57600 for Slurm, and 117600 for the complete pair. Load is within the work budget. Each arm gets 600 additional seconds in the pair capacity check for preflight, accounting and cleanup.

Use the same original pair epoch for every host check. The outer dispatch passes it as --pair-epoch and derives the supervisor timeout from that epoch. The template has no run ID and disables GPU dispatch. The package has not passed target preflight or live Slurm admission.
