"""Bounded full-matrix wall budgets; not a completion-time guarantee."""
PROCESS_SECONDS=21600
NORMAL_CLEANUP_RESERVE=600
CLIENT_MAX_SECONDS=20500
CLIENT_STOP_RESERVE=30
REGULAR_PHASE_MAX=20500
LONG_PHASE_MAX=9000
SHELL_HARD_SECONDS=21720
GPU_SLURM_SECONDS=22200
OBSERVER_SECONDS=22500
KEEPER_SECONDS=23400
def client_deadline(now,service_deadline):return min(now+CLIENT_MAX_SECONDS,service_deadline-CLIENT_STOP_RESERVE)
def phase_timeout(now,deadline,phase_max):
 remaining=deadline-now
 if remaining<=0:raise TimeoutError('Full-matrix client budget exhausted; fail and stop without skipping cases')
 return min(phase_max,remaining)
