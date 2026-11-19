"""One epoch bounds for native workers and their owned supervisor."""
import math
HARD_SECONDS=900
NATIVE_SECONDS=820
SUPERVISOR_STOP_SECONDS=850
TERM_GRACE_SECONDS=10
def remaining(epoch,deadline,now):
    if not all(math.isfinite(x) for x in (epoch,deadline,now)):raise ValueError("nonfinite clock")
    if epoch<=0 or deadline<=epoch or deadline-epoch>HARD_SECONDS:raise ValueError("invalid budget")
    if now<epoch or now>=deadline:raise ValueError("epoch not active")
    return deadline-now
def schedule(epoch,now):
    value={"epoch":epoch,"native_deadline":epoch+NATIVE_SECONDS,"supervisor_stop":epoch+SUPERVISOR_STOP_SECONDS,"deadline":epoch+HARD_SECONDS}
    remaining(epoch,value["deadline"],now)
    assert value["native_deadline"]<value["supervisor_stop"]
    assert value["supervisor_stop"]+2*TERM_GRACE_SECONDS<value["deadline"]
    return value
