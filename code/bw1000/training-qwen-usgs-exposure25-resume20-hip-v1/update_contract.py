"""Differentiate an actual optimizer step from nonzero parameter movement."""
import math


def check_update(step,counters,expected_slots,adapter_changed,learning_rates):
    if (type(step) is not int or not 1<=step<=25 or type(expected_slots) is not int
            or expected_slots<=0 or len(counters)!=expected_slots
            or type(adapter_changed) is not bool or not learning_rates):
        raise ValueError("Incomplete native optimizer update observation")
    if any(type(x) not in (int,float) or not math.isfinite(x) or x!=step for x in counters):
        raise ValueError("Native AdamW counters did not advance to the global step")
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in learning_rates):
        raise ValueError("Invalid observed optimizer learning rate")
    # Warmup may start at LR0. A completed AdamW step can legally leave the
    # adapter unchanged; its slot counters must still advance.
    return dict(global_step=step,optimizer_slot_count=expected_slots,
                zero_learning_rate=all(x==0 for x in learning_rates),adapter_changed=adapter_changed)


def check_complete(changes):
    if len(changes)!=25 or any(type(x) is not bool for x in changes) or not any(changes):
        raise ValueError("The complete 25-update window never changed trainables")
    return dict(updates=25,parameter_changing_updates=sum(changes),parameter_unchanged_updates=25-sum(changes))
