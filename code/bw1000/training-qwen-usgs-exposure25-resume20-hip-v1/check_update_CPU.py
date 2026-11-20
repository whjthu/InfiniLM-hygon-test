"""New regression for zero-LR warmup and real counter advancement; stdlib only."""
from update_contract import check_update,check_complete


def check():
    first=check_update(1,[1.]*504,504,False,[0.,0.])
    assert first["zero_learning_rate"] is True and first["adapter_changed"] is False
    second=check_update(2,[2.]*504,504,True,[1e-4,1e-4])
    assert second["global_step"]==2 and second["adapter_changed"] is True
    complete=check_complete([False]+[True]*24)
    assert complete["parameter_changing_updates"]==24
    rejected=0
    for function,args in ((check_update,(1,[0.]*504,504,False,[0.])),
                          (check_update,(2,[1.]*504,504,True,[1e-4])),
                          (check_complete,([False]*25,))):
        try:
            function(*args)
        except ValueError:
            rejected+=1
    assert rejected==3
    return dict(passed=True,valid_contracts=3,rejected=3,planned_rejections=3,
                legal_zero_LR_first_update=True,real_optimizer_counter_advancement_required=True,
                all25_unchanged_rejected=True,actual_Torch_import=0,actual_optimizer_updates=0,
                actual_GPU=0,actual_target_calls=0)
